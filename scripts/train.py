from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import math

from cs336_basics.transformer.impl import TransformerLM, TransformerLMRope
from cs336_basics.utils.data import get_batch
from cs336_basics.utils.loss import cross_entropy_loss
from cs336_basics.utils.optimizer import AdamW, cosine_lr_schedule, gradient_clipping, MuonWithAuxAdamW
from cs336_basics.utils.checkpointing import save_checkpoint, load_checkpoint

import swanlab

PROJECT_ROOT = Path(__file__).resolve().parents[1]


import time
from typing import Optional

def parse_hms(hms: Optional[str]) -> Optional[int]:
    if not hms:
        return None
    parts = hms.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"cfg.run.time 应为 'HH:MM:SS'，当前为: {hms!r}")
    h, m, s = parts
    h, m, s = int(h), int(m), int(s)
    if not (0 <= m < 60 and 0 <= s < 60 and 0 <= h <= 99):
        raise ValueError(f"非法时间: {h:02d}:{m:02d}:{s:02d}")
    return h * 3600 + m * 60 + s

def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    return torch.device(device)

def swanlab_log(cfg: DictConfig) -> None:
    swanlab.init(
        project="cs336_lm_owt",
        workspace="andantex",
        config=dict(cfg),
    )


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
    return TransformerLMRope(
        vocab_size=vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.theta,
        rotary_dim=16,
        device=device,
        dtype=torch.float32,
    )


def build_optimizer(model: TransformerLM, cfg: DictConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.type == "Muon":
        muon_params = []
        adam_params_scalars = [] # 1D 参数 
        adam_params_special = [] # 2D 特殊参数 (embed, lm_head)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 规则 1: 所有 1D 参数都给 AdamW
            if param.ndim < 2:
                adam_params_scalars.append(param)
            
            # 规则 2: 特定的 2D+ 参数给 AdamW
            elif "embed" in name or "lm_head" in name:
                adam_params_special.append(param)
            
            # 规则 3: 所有其他的 2D+ 参数给 Muon
            else:
                muon_params.append(param)

        # --- 2. 为 AdamW 创建参数组 ---
        # 共享的 AdamW 设置
        adam_settings = dict(
            use_muon=False,
        )
        
        param_groups = [
            dict(params=adam_params_scalars, **adam_settings),
            dict(params=adam_params_special, **adam_settings),
        ]
        
        # --- 3. 为 Muon 创建参数组 ---
        param_groups.append(dict(
            params=muon_params,
            use_muon=True,
        ))
        
        return MuonWithAuxAdamW(param_groups)

    elif cfg.optimizer.type == "AdamW":
        return AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.eps,
        )
        
    else:
        raise ValueError(f"未知的优化器类型: {cfg.optimizer.type}")

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


def eval_all(
    model: "TransformerLM",
    dataset: np.ndarray,             # 支持 np.memmap
    batch_size: int,
    context_length: int,
    device: torch.device,
    *,
    stride: int | None = None,
    pin_memory: bool = True,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    T = context_length
    s = T if stride is None else stride
    s = max(1, min(s, T))  # 1 <= s <= T


    data = torch.from_numpy(dataset)
    if data.numel() < T + 1:
        return float('nan'), float('nan')
    if pin_memory and data.device.type == "cpu":
        data = data.pin_memory()


    Xw = data.unfold(0, T, s)         # [Nx, T], Nx ~= floor((L - T)/s) + 1
    Yw = data[1:].unfold(0, T, s)     # [Ny, T], Ny ~= floor((L - 1 - T)/s) + 1
    num_rows = min(Xw.size(0), Yw.size(0))  

    non_blocking = pin_memory and (device.type == "cuda")

    with torch.no_grad():
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            if end <= start:
                break


            X_cpu_full = Xw[start:end].contiguous()  # [B, T]
            Y_cpu_full = Yw[start:end].contiguous()  # [B, T]
            B = X_cpu_full.size(0)

            X_cpu = X_cpu_full[:, T - s:]
            Y_cpu = Y_cpu_full[:, T - s:]

            X = X_cpu.to(device, non_blocking=non_blocking)
            Y = Y_cpu.to(device, non_blocking=non_blocking)

            logits = model(X)                 
            loss = cross_entropy_loss(logits, Y)  

            total_loss   += loss.item() * (B * s)  
            total_tokens += (B * s)

    if total_tokens == 0:
        return float('nan'), float('nan')

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return float(avg_loss), float(ppl)

def eval_single(model: TransformerLM, dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device) -> float:
    model.eval()
    inputs, targets = get_batch(dataset, batch_size, context_length, str(device))

    with torch.no_grad():
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
    return loss.item()


@hydra.main(version_base=None, config_path="conf", config_name="run")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.run.device)
    swanlab_log(cfg)
    train_tokens, valid_tokens = load_datasets(cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)

    checkpoint_dir = PROJECT_ROOT / cfg.run.checkpointing.path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # NEW: 确保目录存在

    if cfg.run.load_checkpoint:
        start_iter = load_checkpoint(
            src=checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
        )
    else:
        start_iter = 0

    # 既有 token 目标；时间为上限，二者谁先满足谁触发停止
    total_iters = cfg.run.total_token // (cfg.run.batch_size * cfg.model.context_length)

    # NEW: 解析时间上限（秒）
    time_budget_s = None
    try:
        # DictConfig 支持 .get
        time_str = cfg.run.get("time", None)
    except Exception:
        time_str = None
    time_budget_s = parse_hms(time_str)  # 可为 None

    start_wall = time.time()             # NEW
    last_iter_done = start_iter - 1      # NEW: 记录“最后完成的迭代编号”

    # 用 try/finally 确保必存
    try:
        for iteration in range(start_iter, total_iters):
            # 若设定了时间上限且已超时，则提前停止
            if time_budget_s is not None:
                elapsed = time.time() - start_wall
                if elapsed >= time_budget_s:
                    print(f"⏱️ 触达时间上限 {time_str}，在迭代 {iteration} 提前停止。")
                    break

            if cfg.run.use_cosine:
                update_lr(optimizer, cfg, iteration, total_iters)

            inputs, targets = get_batch(
                train_tokens, cfg.run.batch_size, cfg.model.context_length, str(device)
            )
            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()

            if cfg.run.use_grad_clip:
                clip_gradients(model, cfg)
            optimizer.step()

            train_loss = loss.item()
            val_loss = eval_single(
                model,
                valid_tokens,
                cfg.run.batch_size,
                cfg.model.context_length,
                device,
            )
            swanlab.log({
                "train/loss": train_loss,
                "val/loss_single": val_loss,
            }, step=iteration)

            if cfg.run.eval_interval > 0 and (iteration + 1) % cfg.run.eval_interval == 0:
                val_loss_all, perplexity = eval_all(
                    model,
                    valid_tokens,
                    cfg.run.batch_size,
                    cfg.model.context_length,
                    device,
                )
                swanlab.log({
                    "val/loss_all": val_loss_all,
                    "val/perplexity": perplexity,
                }, step=iteration)

            if cfg.run.checkpointing.save_interval > 0 and (iteration + 1) % cfg.run.checkpointing.save_interval == 0:
                save_checkpoint(
                    out=checkpoint_dir / f"iter_{iteration + 1}.pt",
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                )
                # 同步一份 latest.pt 方便 resume（可选但强烈建议）
                save_checkpoint(
                    out=checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                )

            last_iter_done = iteration  # NEW: 记录已完成的迭代
    finally:
        # NEW: 退出前必存（无论是正常结束、时间触发、异常中断）
        final_tag = (last_iter_done + 1)  # 人类友好地用“已完成迭代数”计数
        final_path_iter = checkpoint_dir / f"final_iter_{final_tag}.pt"
        final_path_latest = checkpoint_dir / "latest.pt"

        save_checkpoint(
            out=final_path_iter,
            model=model,
            optimizer=optimizer,
            iteration=last_iter_done,
        )
        save_checkpoint(
            out=final_path_latest,
            model=model,
            optimizer=optimizer,
            iteration=last_iter_done,
        )
        print(f"✅ 已保存最终检查点：{final_path_iter.name} 与 latest.pt")

  

if __name__ == "__main__":
    main()