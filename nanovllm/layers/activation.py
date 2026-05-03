import torch
from torch import nn
import torch.nn.functional as F

_TORCH_COMPILE = torch.cuda.get_device_capability() >= (7, 0)
_conditional_compile = torch.compile if _TORCH_COMPILE else lambda f: f


class SiluAndMul(nn.Module):

    @_conditional_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
