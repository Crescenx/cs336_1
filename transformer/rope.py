import torch
from torch import nn
import einx

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_exp = torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
        inv_freqs = 1.0 / (theta ** freqs_exp)
        theta_matrix = torch.einsum("i,j->i j", positions, inv_freqs)
        self.register_buffer("cos_cached", torch.cos(theta_matrix).to(device), persistent=False) # (max_seq_len, d_k/2)
        self.register_buffer("sin_cached", torch.sin(theta_matrix).to(device), persistent=False) # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (... seq_len d_k)
        # token_positions: (... seq_len)
        cos = self.cos_cached[token_positions] # (seq_len, d_k/2)
        sin = self.sin_cached[token_positions] # (seq_len, d_k/2)
        real, imag = x[..., ::2], x[..., 1::2] # (... seq_len d_k/2)
        rotated_real = real * cos - imag * sin # (... seq_len d_k/2)
        rotated_imag = real * sin + imag * cos # (... seq_len d_k/2)
        return torch.stack((rotated_real, rotated_imag), dim=-1).flatten(-2) # (... seq_len d_k)


