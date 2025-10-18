import torch
from torch import nn
import einx

from transformer.rope import RoPE

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x

def scaled_dot_product_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        query: Float[Tensor, " batch_size ... seq_len d_k"]: Query tensor.
        key: Float[Tensor, "batch_size ... seq_len d_k"]: Key tensor.
        value: Float[Tensor, "batch_size ... seq_len d_v"]: Value tensor.
        mask: Optional[Bool[Tensor, "seq_len seq_len"]]: Optional mask tensor. If provided,

    Returns:
        Float[Tensor, "batchsize ... seq_len d_v"]: Output tensor after applying attention.
    """ 
    d_k = query.shape[-1]
    scores = einx.dot("... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k", query, key) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = einx.dot("... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v", attn_weights, value)
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.output_proj = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.rope = rope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.q_proj, std=0.02)
        nn.init.trunc_normal_(self.k_proj, std=0.02)
        nn.init.trunc_normal_(self.v_proj, std=0.02)
        nn.init.trunc_normal_(self.output_proj, std=0.02)

    def forward(self, x: torch.Tensor, causual: bool=False) -> torch.Tensor:
        seq_len = x.shape[-2]

        W_qkv = torch.stack((self.q_proj, self.k_proj, self.v_proj), dim=0) # (3, d_model, d_model)
        QKV = einx.rearrange(
            "... seq_len three (h d_k) -> ... three h seq_len d_k",
            einx.dot("... seq_len d_model, three hd_k d_model -> ... seq_len three hd_k", x, W_qkv),
            h=self.num_heads,
        )
        Q, K, V = QKV.unbind(dim=-4) # each is (... num_heads seq_len d_k)
        

        # RoPE Q, K
        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal mask attn
        if causual:
            mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
            out = scaled_dot_product_attn(Q, K, V, mask=mask)
        out = einx.rearrange("... num_heads seq_len d_k -> ... seq_len (num_heads d_k)", out)
        out = einx.dot("... seq_len d_v, d_model d_v -> ... seq_len d_model", out, self.output_proj)
        return out
        
