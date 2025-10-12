import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
        