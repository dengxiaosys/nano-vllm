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


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(bs, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bs, num_kv_heads * n_rep, seq_len, head_dim)


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
        ki = _repeat_kv(ki, n_rep)
        vi = _repeat_kv(vi, n_rep)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=True)
        outputs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _decode_sdpa(q, k_cache, v_cache, context_lens, block_tables, scale, num_heads, num_kv_heads, head_dim, block_size):
    n_rep = num_heads // num_kv_heads
    num_seqs = q.shape[0]
    outputs = []
    for i in range(num_seqs):
        qi = q[i].unsqueeze(1).unsqueeze(0)
        seq_len = context_lens[i].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_tables[i, :num_blocks]
        ki = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len].transpose(0, 1).unsqueeze(0)
        vi = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len].transpose(0, 1).unsqueeze(0)
        ki = _repeat_kv(ki, n_rep)
        vi = _repeat_kv(vi, n_rep)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        outputs.append(oi.squeeze(0).transpose(0, 1).squeeze(0))
    return torch.stack(outputs, dim=0)


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
