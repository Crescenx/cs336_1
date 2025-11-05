import torch
from torch import nn
import einx

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_exp = torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
        inv_freqs = 1.0 / (theta ** freqs_exp)
        theta_matrix = einx.multiply("i,j->i j", positions, inv_freqs)
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


class RoPEBuf(nn.Module):
    def __init__(self, 
                 rotary_dim: int, 
                 max_seq_len: int, 
                 theta: float = 10000.0, 
                 device=None, 
                 dtype=None):
        super().__init__()
        
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs_exp = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim
        inv_freqs = 1.0 / (theta ** freqs_exp)
        theta_matrix = einx.multiply("i,j->i j", positions, inv_freqs)
        
        self.register_buffer("cos_cached", torch.cos(theta_matrix).to(dtype), persistent=False)
        self.register_buffer("sin_cached", torch.sin(theta_matrix).to(dtype), persistent=False)

    def forward(self, token_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cos_cached[token_positions]
        s = self.sin_cached[token_positions]
        return c, s
    
def apply_rotary_pos_emb(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:

    rotary_dim = cos.shape[-1] * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    
    cos = einx.rearrange("(1 1 s) d -> 1 1 s d", cos)
    sin = einx.rearrange("(1 1 s) d -> 1 1 s d", sin)
    
    real, imag = x_rot[..., ::2], x_rot[..., 1::2]
    rotated_real = real * cos - imag * sin
    rotated_imag = real * sin + imag * cos

    x_rotated = torch.stack((rotated_real, rotated_imag), dim=-1).flatten(-2)
    
    return torch.cat((x_rotated, x_pass), dim=-1)