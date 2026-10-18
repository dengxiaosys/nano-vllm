from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False  # 本轮是否为 prefill；False 时走每序列一个查询 token 的 decode 路径
    cu_seqlens_q: torch.Tensor | None = None  # prefill 各序列查询长度的累积和，形状 [batch_size + 1]、首项为 0；命中前缀时只计待计算后缀
    cu_seqlens_k: torch.Tensor | None = None  # prefill 各序列完整 KV 长度的累积和，形状 [batch_size + 1]；包含已缓存前缀，分页时不是物理存储偏移
    max_seqlen_q: int = 0  # prefill 批次中最长的查询序列长度，供 FlashAttention 内核规划计算
    max_seqlen_k: int = 0  # prefill 批次中最长的完整 KV 序列长度，包含已缓存前缀
    slot_mapping: torch.Tensor | None = None  # 本轮每个输入 token 的 KV 写入槽位，形状 [num_input_tokens]；槽位 = 物理块 ID * block_size + 块内偏移，-1 表示跳过写入
    context_lens: torch.Tensor | None = None  # decode 每个序列的有效 KV 长度，形状 [batch_size]；包含本轮写入 KV 的当前 token
    block_tables: torch.Tensor | None = None  # 每序列的逻辑块到物理块 ID 映射，形状 [batch_size, max_num_blocks]、以 -1 补齐；用于 decode 或命中前缀的 prefill

_CONTEXT = Context()  # 当前进程共享的本轮前向上下文，供各层读取；set_context 替换，reset_context 重置

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
