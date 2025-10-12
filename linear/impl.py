import torch.nn as nn
import torch
import einx

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.W, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... in , out in-> ... out", x, self.W)