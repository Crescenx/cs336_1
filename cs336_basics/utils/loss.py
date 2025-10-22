import torch
import einx

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (batch... vocab_size)
    # targets: (batch...)
    max_logits = logits.max(dim=-1, keepdim=True).values
    downcasted_logits = logits - max_logits
    logsumexp = torch.log(einx.sum("b... [vocab_size] -> b...", torch.exp(downcasted_logits)))
    expanded_targets = targets.unsqueeze(-1)
    target_logit = einx.get_at("b... [vocab_size], b... [1] -> b...", downcasted_logits, expanded_targets)
    loss = logsumexp - target_logit
    return einx.mean("[b...]", loss)
