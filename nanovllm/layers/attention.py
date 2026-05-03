import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

_USE_FLASH_ATTN = torch.cuda.get_device_capability() >= (8, 0)

if _USE_FLASH_ATTN:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _prefill_sdpa(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, num_heads, num_kv_heads):
    n_rep = num_heads // num_kv_heads
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        ki = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        vi = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            ki = ki.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(1, num_heads, -1, ki.shape[-1])
            vi = vi.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(1, num_heads, -1, vi.shape[-1])
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=True)
        outputs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _decode_sdpa(q, k_cache, v_cache, context_lens, block_tables, scale, num_heads, num_kv_heads, head_dim, block_size):
    batch_size = q.shape[0]
    max_num_blocks = block_tables.shape[1]
    max_seq_len = max_num_blocks * block_size
    n_rep = num_heads // num_kv_heads

    block_ids = block_tables.clamp(min=0)
    k_paged = k_cache[block_ids]
    v_paged = v_cache[block_ids]

    k_flat = k_paged.reshape(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_flat = v_paged.reshape(batch_size, max_seq_len, num_kv_heads, head_dim)

    arange = torch.arange(max_seq_len, device=q.device)
    valid_mask = arange.unsqueeze(0) < context_lens.unsqueeze(1)

    valid_kv = valid_mask.unsqueeze(-1).unsqueeze(-1).to(k_flat.dtype)
    k_flat = k_flat * valid_kv
    v_flat = v_flat * valid_kv

    if n_rep > 1:
        k_flat = k_flat.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch_size, max_seq_len, num_heads, head_dim)
        v_flat = v_flat.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch_size, max_seq_len, num_heads, head_dim)

    q_4d = q.unsqueeze(2)
    k_4d = k_flat.transpose(1, 2)
    v_4d = v_flat.transpose(1, 2)

    attn_mask = torch.zeros(batch_size, 1, 1, max_seq_len, dtype=q.dtype, device=q.device)
    attn_mask = attn_mask.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), torch.finfo(q.dtype).min)

    o = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, attn_mask=attn_mask, scale=scale)
    return o.squeeze(2)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if _USE_FLASH_ATTN:
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        else:
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = _prefill_sdpa(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k,
                                  context.max_seqlen_q, context.max_seqlen_k, self.scale,
                                  self.num_heads, self.num_kv_heads)
            else:
                block_size = k_cache.shape[1]
                o = _decode_sdpa(q, k_cache, v_cache, context.context_lens, context.block_tables,
                                 self.scale, self.num_heads, self.num_kv_heads, self.head_dim, block_size)
        return o
