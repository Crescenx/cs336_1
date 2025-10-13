import torch
import torch.nn as nn
import einx

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast to float32 for numerical stability
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x ** 2
        rms_x = torch.sqrt(einx.mean("... [d_model]", x_squared, keepdims=True) + self.eps)
        result = x / rms_x * self.gain
        return result.to(input_dtype)