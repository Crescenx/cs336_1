import torch
from torch import nn

from cs336_basics.transformer.attn import MultiHeadSelfAttention
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.swiglu import SwiGLU
from cs336_basics.transformer.rope import RoPE

from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()

        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        # self.rope = None
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (... seq_len d_model)
        attn_out = self.attn(self.ln1(x), causual=True)
        x = attn_out + x
        ff_out = self.ffn(self.ln2(x))
        x = ff_out + x
        return x
    

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size seq_len)
        x = self.token_embeddings(input)  # (batch_size seq_len d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (batch_size seq_len vocab_size)
        return logits