import torch
from torch import nn

from cs336_basics.transformer.linear import Linear

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.reset_parameters()
        self.silu = SiLU()
    
    def reset_parameters(self) -> None:
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.silu((self.w1(x)))
        x_3 = self.w3(x)
        return self.w2(x_1 * x_3)