from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import einx

from cs336_basics.transformer.impl import TransformerLM
from cs336_basics.utils.data import get_batch
from cs336_basics.utils.loss import cross_entropy_loss
from cs336_basics.utils.optimizer import AdamW, cosine_lr_schedule, gradient_clipping
from cs336_basics.utils.checkpointing import save_checkpoint, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    return torch.device(device)


def load_datasets(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.run.type == "tiny":
        data_dir = PROJECT_ROOT / "data" / "tokenized"
        train_path = data_dir / "tinystories_train.npy"
        valid_path = data_dir / "tinystories_valid.npy"
    elif cfg.run.type == "owt":
        data_dir = PROJECT_ROOT / "data" / "tokenized"
        train_path = data_dir / "owt_train.npy"
        valid_path = data_dir / "owt_valid.npy"

    train_data = np.load(train_path, mmap_mode="r").astype(np.long)
    valid_data = np.load(valid_path, mmap_mode="r").astype(np.long)
    return train_data, valid_data


def build_model(cfg: DictConfig, device: torch.device) -> TransformerLM:
    if cfg.run.type == "tiny":
        vocab_size = cfg.model.vocab_size.tiny
    elif cfg.run.type == "owt":
        vocab_size = cfg.model.vocab_size.owt
    else:
        raise ValueError(f"Unknown run type: {cfg.type}")
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.theta,
        device=device,
        dtype=torch.float32,
    )


def build_optimizer(model: TransformerLM, cfg: DictConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
    )


def update_lr(optimizer: AdamW, cfg: DictConfig, iteration: int, total_iters: int) -> None:
    lr = cosine_lr_schedule(
        iteration,
        max_learning_rate=cfg.optimizer.cosine.max_lr,
        min_learning_rate=cfg.optimizer.cosine.min_lr,
        warmup_iters=int(cfg.optimizer.cosine.warmup * total_iters),
        cosine_cycle_iters=int(cfg.optimizer.cosine.cycles * total_iters),
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def clip_gradients(model: TransformerLM, cfg: DictConfig) -> None:
    gradient_clipping(model.parameters(), cfg.optimizer.grad_clip)


def evaluate(model: TransformerLM, dataset: np.ndarray, context_length: int, batch_size: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    N = len(dataset)
    indices = np.arange(0, N - context_length)

    with torch.no_grad():
        for start_idx in range(0, len(indices), batch_size):
            batch_inds = indices[start_idx : start_idx + batch_size]
            actual_b = len(batch_inds)
            if actual_b == 0:
                continue

            X = torch.stack(
                [torch.tensor(dataset[i : i + context_length], device=device) for i in batch_inds]
            )
            Y = torch.stack(
                [torch.tensor(dataset[i + 1 : i + context_length + 1], device=device) for i in batch_inds]
            )

            logits = model(X)  # shape (actual_b, context_length, vocab_size)

            loss = cross_entropy_loss(logits, Y)

            tokens_in_batch = actual_b * context_length
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch

    if total_tokens == 0:
        return float('nan'), float('nan')

    average_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(average_loss)))

    return average_loss, perplexity



@hydra.main(version_base=None, config_path="conf", config_name="run")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.run.device)
    train_tokens, valid_tokens = load_datasets(cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)

    checkpoint_dir = PROJECT_ROOT / cfg.run.checkpointing.path
    if cfg.run.load_checkpoint:
        start_iter = load_checkpoint(
            src=checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
        )
    else:
        start_iter = 0

    total_iters = cfg.run.total_token // (cfg.run.batch_size * cfg.model.context_length)

    for iteration in range(start_iter, total_iters):
        if cfg.run.use_cosine:
            update_lr(optimizer, cfg, iteration, total_iters)

        inputs, targets = get_batch(train_tokens, cfg.run.batch_size, cfg.model.context_length, str(device))
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        if cfg.run.use_grad_clip:
            clip_gradients(model, cfg)
        optimizer.step()

        if (iteration + 1) % cfg.run.eval_interval == 0:
            val_loss, perplexity = evaluate(
                model,
                valid_tokens,
                cfg.model.context_length,
                cfg.run.batch_size,
                device,
            )
            print(f"Validation loss: {val_loss}, Perplexity: {perplexity}")

        if (iteration + 1) % cfg.run.checkpointing.save_interval == 0:
            save_checkpoint(
                dst=checkpoint_dir / f"iter_{iteration + 1}.pt",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
            )
  

if __name__ == "__main__":
    main()