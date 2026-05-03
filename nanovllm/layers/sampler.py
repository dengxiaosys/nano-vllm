import torch
from torch import nn

_TORCH_COMPILE = torch.cuda.get_device_capability() >= (7, 0)
_conditional_compile = torch.compile if _TORCH_COMPILE else lambda f: f


class Sampler(nn.Module):

    @_conditional_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
