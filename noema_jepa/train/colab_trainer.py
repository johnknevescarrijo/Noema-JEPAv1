"""
ColabTrainer: High-Performance Lightweight Trainer for Google Colab.

Engineered with:
- AMP (Automatic Mixed Precision: fp16 on T4 / bf16 on L4/A100) + GradScaler.
- Native step-by-step VarianceWatchdog integration to prevent representation collapse.
- Low-overhead memory management, optional torch.compile, and throughput tracking.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from noema_jepa.config import NoemaConfig
from noema_jepa.losses.vicreg import NoemaWorldModelLoss
from noema_jepa.models.noema_jepa import NoemaJEPA
from noema_jepa.monitoring.variance_watchdog import VarianceWatchdog, WatchdogReport


class ColabTrainer:
    """
    Lightweight Trainer tailored for Google Colab GPU runtimes.
    Coordinates forward/backward passes, AMP scaling, and the VarianceWatchdog.
    """

    def __init__(
        self,
        model: NoemaJEPA,
        loss_fn: Optional[NoemaWorldModelLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        watchdog: Optional[VarianceWatchdog] = None,
        config: Optional[NoemaConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config or getattr(model, "config", NoemaConfig())
        cfg = self.config

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Enable TF32 precision for Tensor Cores on Ampere/Ada/Hopper (A100 / L4)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = model.to(self.device)
        self.loss_fn = (loss_fn or NoemaWorldModelLoss(cfg)).to(self.device)
        self.watchdog = watchdog or VarianceWatchdog(cfg)

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Optional torch.compile for PyTorch 2.0+ (gracefully skipped on CPU or Triton absence)
        if cfg.use_compile and hasattr(torch, "compile"):
            if self.device.type == "cuda":
                print("[ColabTrainer] Compiling model with torch.compile()...")
                try:
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print(f"[ColabTrainer Warning] torch.compile failed ({e}), continuing in eager mode.")
            else:
                # Graceful CPU fallback (e.g. during local tests)
                pass

        # Mixed Precision Setup
        self.use_amp = cfg.use_amp and (self.device.type == "cuda")
        amp_dtype_str = cfg.amp_dtype.lower()
        if amp_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # GradScaler is only enabled for float16. In bfloat16 (native A100/L4),
        # dynamic range matches float32, making GradScaler unnecessary overhead.
        scaler_enabled = self.use_amp and (self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        self.global_step = 0

    def train_step(
        self,
        batch: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> Tuple[float, Dict[str, float], WatchdogReport]:
        """
        Executes a single forward/backward training step with AMP and Watchdog update.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Unpack batch
        if isinstance(batch, dict):
            images = batch["images"].to(self.device, non_blocking=True)
            think_embs = batch.get("think_embeddings")
            if think_embs is not None:
                think_embs = think_embs.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(self.device, non_blocking=True)
            think_embs = batch[1].to(self.device, non_blocking=True) if len(batch) > 1 else None
        else:
            images = batch.to(self.device, non_blocking=True)
            think_embs = None

        # Fetch active modulation scale from watchdog
        mod_scale = self.watchdog.get_modulation_scale()

        # Mixed-precision forward pass
        device_type = self.device.type
        with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                images=images,
                think_embeddings=think_embs,
                modulation_scale=mod_scale,
            )
            loss, metrics = self.loss_fn(outputs)

        # Scaled backward pass
        self.scaler.scale(loss).backward()

        if self.config.grad_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        # Real-time Variance Watchdog verification and dynamic feedback
        report = self.watchdog.step(
            metrics=metrics,
            model=self.model,
            loss_fn=self.loss_fn,
        )

        self.global_step += 1
        return loss.item(), metrics, report

    def train_epoch(self, dataloader: DataLoader, epoch: int = 1) -> Dict[str, float]:
        """
        Trains the model for one full epoch over the dataloader.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        num_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for batch in pbar:
            batch_size = (
                batch["images"].size(0) if isinstance(batch, dict)
                else (batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0))
            )
            num_samples += batch_size

            loss, metrics, report = self.train_step(batch)
            total_loss += loss

            pbar.set_postfix({
                "loss": f"{loss:.3f}",
                "pred": f"{metrics['loss_pred']:.3f}",
                "intra_std": f"{metrics['metric_intra_scene_std']:.3f}",
                "scale": f"{report.current_modulation_scale:.2f}",
                "status": report.status,
            })

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(len(dataloader), 1)
        throughput = num_samples / max(elapsed, 1e-4)

        return {
            "avg_loss": avg_loss,
            "throughput_samples_per_sec": throughput,
            "total_steps": self.global_step,
            "watchdog_warnings": self.watchdog.collapse_warnings_total,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Saves model weights, optimizer, and watchdog state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        torch.save({
            "step": self.global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "watchdog_summary": self.watchdog.get_summary(),
            "config": self.config,
        }, str(path))
        print(f"[ColabTrainer] Checkpoint saved at {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Loads checkpoint into model and optimizer."""
        checkpoint = torch.load(str(path), map_location=self.device)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = checkpoint.get("step", 0)
        print(f"[ColabTrainer] Checkpoint loaded from {path} (resuming at step {self.global_step})")
