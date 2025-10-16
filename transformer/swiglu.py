import torch
from torch import nn
import einx

from linear.impl import Linear

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.reset_parameters()
        self.silu = SiLU()
    
    def reset_parameters(self) -> None:
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.silu((self.W1(x)))
        x_3 = self.W3(x)
        return self.W2(x_1 * x_3)