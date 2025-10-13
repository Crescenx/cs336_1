import torch
from torch import nn
import einx

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.reset_parameters()
        self.silu = SiLU()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.W1, std=0.02)
        nn.init.trunc_normal_(self.W2, std=0.02)
        nn.init.trunc_normal_(self.W3, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.silu(einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.W1))
        x_3 = einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.W3)
        return einx.dot("... d_ff, d_model d_ff -> ... d_model", x_1 * x_3, self.W2)