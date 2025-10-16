import torch
from torch import nn

from transformer.attn import MultiHeadSelfAttention
from transformer.rmsnorm import RMSNorm
from transformer.swiglu import SwiGLU
from transformer.rope import RoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()

        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (... seq_len d_model)
        attn_out = self.attn(self.norm1(x), causual=True)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x