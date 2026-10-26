"""
ColabTrainer: High-Performance Lightweight Trainer for Google Colab.

Engineered with:
- AMP (Automatic Mixed Precision: fp16 on T4 / bf16 on L4/A100) + GradScaler.
- Native step-by-step VarianceWatchdog integration to prevent representation collapse.
- Low-overhead memory management, optional torch.compile, and throughput tracking.
"""

import math
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

        # Scheduler (automatic Cosine Annealing with Warmup if enabled)
        if scheduler is None and getattr(cfg, "use_scheduler", False):
            warmup = max(1, getattr(cfg, "warmup_steps", 50))
            total = max(warmup + 1, getattr(cfg, "total_steps", 1000))

            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                progress = min(1.0, float(step - warmup) / float(max(1, total - warmup)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_ratio = cfg.min_lr / max(cfg.learning_rate, 1e-8)
                return min_ratio + (1.0 - min_ratio) * cosine

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = scheduler

        # Optional torch.compile for PyTorch 2.0+ (gracefully skipped on CPU or Triton absence)
        self.is_compiled = False
        if cfg.use_compile and hasattr(torch, "compile"):
            if self.device.type == "cuda":
                print("[ColabTrainer] Compiling model with torch.compile()...")
                try:
                    self.model = torch.compile(self.model)
                    self.is_compiled = True
                except Exception as e:
                    print(f"[ColabTrainer Warning] torch.compile failed ({e}), continuing in eager mode.")
                    self.is_compiled = False
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
        batch: Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...]],
    ) -> Tuple[float, Dict[str, float], WatchdogReport]:
        """
        Executes a single forward/backward training step with AMP and Watchdog update.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Unpack batch safely supporting 'images' or 'image', 'think_embeddings' or 'think_emb'
        if isinstance(batch, dict):
            images = batch.get("images")
            if images is None:
                images = batch.get("image")
            if images is None:
                raise KeyError("Batch dictionary contains neither 'images' nor 'image' key.")
            images = images.to(self.device, non_blocking=True)

            think_embs = batch.get("think_embeddings")
            if think_embs is None:
                think_embs = batch.get("think_emb")
            if think_embs is not None and isinstance(think_embs, torch.Tensor) and think_embs.numel() > 0:
                think_embs = think_embs.to(self.device, non_blocking=True)
            else:
                think_embs = None
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(self.device, non_blocking=True)
            think_embs = (
                batch[1].to(self.device, non_blocking=True)
                if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].numel() > 0
                else None
            )
        else:
            images = batch.to(self.device, non_blocking=True)
            think_embs = None

        # Fetch active modulation scale from watchdog
        mod_scale = self.watchdog.get_modulation_scale()

        # Mixed-precision forward pass with graceful compiled-mode fallback
        device_type = self.device.type
        try:
            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(
                    images=images,
                    think_embeddings=think_embs,
                    modulation_scale=mod_scale,
                )
                loss, metrics = self.loss_fn(outputs)
        except Exception as fwd_exc:
            if self.is_compiled:
                print(f"[ColabTrainer Warning] Compiled forward pass encountered error ({fwd_exc}). Falling back to eager execution.")
                self.model = getattr(self.model, "_orig_mod", self.model)
                self.is_compiled = False
                with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        images=images,
                        think_embeddings=think_embs,
                        modulation_scale=mod_scale,
                    )
                    loss, metrics = self.loss_fn(outputs)
            else:
                raise fwd_exc

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
        Returns complete metrics dictionary with all standard benchmark keys.
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_vicreg_loss = 0.0
        total_intra_std = 0.0
        num_batches = 0
        num_samples = 0
        start_time = time.time()

        if len(dataloader) == 0:
            return {
                "avg_loss": 0.0,
                "loss": 0.0,
                "loss_pred": 0.0,
                "pred_loss": 0.0,
                "loss_vicreg": 0.0,
                "vicreg_loss": 0.0,
                "intra_std": 0.0,
                "throughput_samples_per_sec": 0.0,
                "throughput": 0.0,
                "total_steps": self.global_step,
                "watchdog_warnings": self.watchdog.collapse_warnings_total,
            }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for batch in pbar:
            if isinstance(batch, dict):
                img = batch.get("images") if batch.get("images") is not None else batch.get("image")
                batch_size = img.size(0) if img is not None else 1
            elif isinstance(batch, (list, tuple)):
                batch_size = batch[0].size(0)
            else:
                batch_size = batch.size(0)

            num_samples += batch_size

            loss, metrics, report = self.train_step(batch)
            total_loss += loss

            pred_l = metrics.get("loss_pred", metrics.get("pred_loss", 0.0))
            vic_l = metrics.get("loss_vicreg", metrics.get("vicreg_loss", 0.0))
            intra_s = metrics.get("metric_intra_scene_std", metrics.get("intra_std", 0.0))

            total_pred_loss += pred_l
            total_vicreg_loss += vic_l
            total_intra_std += intra_s
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss:.3f}",
                "pred": f"{pred_l:.3f}",
                "intra_std": f"{intra_s:.3f}",
                "scale": f"{report.current_modulation_scale:.2f}",
                "status": report.status,
            })

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        avg_pred = total_pred_loss / max(num_batches, 1)
        avg_vicreg = total_vicreg_loss / max(num_batches, 1)
        avg_intra = total_intra_std / max(num_batches, 1)
        throughput = num_samples / max(elapsed, 1e-4)

        return {
            "avg_loss": avg_loss,
            "loss": avg_loss,
            "loss_pred": avg_pred,
            "pred_loss": avg_pred,
            "loss_vicreg": avg_vicreg,
            "vicreg_loss": avg_vicreg,
            "intra_std": avg_intra,
            "throughput_samples_per_sec": throughput,
            "throughput": throughput,
            "total_steps": self.global_step,
            "watchdog_warnings": self.watchdog.collapse_warnings_total,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Saves model weights, optimizer, and watchdog state."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        torch.save({
            "step": self.global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "watchdog_summary": self.watchdog.get_summary(),
            "config_dict": dataclasses.asdict(self.config) if dataclasses.is_dataclass(self.config) else None,
        }, str(path))
        print(f"[ColabTrainer] Checkpoint saved at {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Loads checkpoint into model and optimizer."""
        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as opt_err:
                print(f"[ColabTrainer Warning] Optimizer state loading skipped: {opt_err}")
        self.global_step = checkpoint.get("step", 0)
        print(f"[ColabTrainer] Checkpoint loaded from {path} (resuming at step {self.global_step})")
