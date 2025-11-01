from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import einx

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # Prepare
                t = state.get('step', 1)
                grad = p.grad.data
                m = state.get('first', torch.zeros_like(grad))
                v = state.get('second', torch.zeros_like(grad))
                rate = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Update
                state['first'] = beta1 * m + (1 - beta1) * grad
                state['second'] = beta2 * v + (1 - beta2) * grad ** 2
                p.data.add_(-rate * state['first'] / (torch.sqrt(state['second']) + eps))

                # Ending
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - lr * weight_decay)
                state['step'] = t + 1

        return loss

def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        return min_learning_rate
    

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)



def zeropower_via_newtonschulz5(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert M.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # To get smaller singular value
    X = M.bfloat16()
    if M.shape[-2] > M.shape[-1]:
        X = einx.rearrange('... i j -> ... j i', X)

    # Iteration
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = einx.dot('... i [j] , ... k [j] -> ... i k', X, X) # X@X.mT
        B = b * A + c * einx.dot('... i j , ... j k -> ... i k', A, A)
        X = a * X + einx.dot('... i j , ... j k -> ... i k', B, X)

    if M.shape[-2] > M.shape[-1]:
        X = einx.rearrange('... i j -> ... j i', X)
    
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, n_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    # apply Nesterov-style momentum
    update = grad.lerp(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update, steps=n_steps)
    update *= math.sqrt(max(grad.shape[-1], grad.shape[-2])) * 0.2
    # update *= math.sqrt(max(1, grad.shape[-2]/grad.shape[-1]))
    return update

def adam_update(
        grad: torch.Tensor, 
        buf1: torch.Tensor, 
        buf2: torch.Tensor, 
        step: int, 
        betas: tuple[float, float],
        eps: float = 1e-8,
    ) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class MuonWithAuxAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss