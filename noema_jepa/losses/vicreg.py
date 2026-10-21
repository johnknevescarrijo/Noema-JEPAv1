"""
VICReg and Noema World Model Loss functions.

Engineered with fp32 stabilization for Google Colab AMP (fp16 on T4 / bf16 on L4/A100).
Decouples:
1. Object prediction loss (Smooth L1 / MSE)
2. VICReg regularization (Invariance, Variance hinge, Covariance decorrelation)
3. Intra-scene object variance penalty (anti-FiLM collapse)
4. ThinkJEPA semantic alignment (Cosine / InfoNCE)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from noema_jepa.config import NoemaConfig


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extracts off-diagonal elements of a 2D square matrix."""
    n, m = x.shape
    assert n == m, "Matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """
    Standard Variance-Invariance-Covariance Regularization loss (Bardes et al., 2022).
    Critical Colab Optimization:
    Always computes Variance and Covariance in float32 to prevent fp16 underflows/NaNs.
    """

    def __init__(
        self,
        sim_coeff: float = 1.0,
        var_coeff: float = 1.0,
        cov_coeff: float = 0.04,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_pred: Projected latents from predicted slots [N_samples, D].
            z_target: Projected latents from target slots [N_samples, D].

        Returns:
            Tuple of (total_vicreg_loss, metrics_dict).
        """
        # 1. Invariance (MSE loss)
        sim_loss = F.mse_loss(z_pred, z_target)

        # 2. Force float32 precision for Variance & Covariance to protect against AMP fp16 underflow
        z_pred_f32 = z_pred.float()
        z_target_f32 = z_target.float()
        N, D = z_pred_f32.shape

        # Batch variance (hinge loss: force std >= gamma)
        # unbiased=False prevents NaN and degrees of freedom warnings when N=1
        std_pred = torch.sqrt(z_pred_f32.var(dim=0, unbiased=False) + self.eps)
        std_target = torch.sqrt(z_target_f32.var(dim=0, unbiased=False) + self.eps)

        var_loss = (
            torch.mean(F.relu(self.gamma - std_pred))
            + torch.mean(F.relu(self.gamma - std_target))
        ) / 2.0

        # Batch covariance (force off-diagonal decorrelation)
        z_pred_centered = z_pred_f32 - z_pred_f32.mean(dim=0)
        z_target_centered = z_target_f32 - z_target_f32.mean(dim=0)

        cov_pred = (z_pred_centered.T @ z_pred_centered) / max(N - 1, 1)
        cov_target = (z_target_centered.T @ z_target_centered) / max(N - 1, 1)

        cov_loss = (
            off_diagonal(cov_pred).pow(2).sum() / D
            + off_diagonal(cov_target).pow(2).sum() / D
        )

        total_vicreg = (
            self.sim_coeff * sim_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        metrics = {
            "vicreg_sim_loss": sim_loss.item(),
            "vicreg_var_loss": var_loss.item(),
            "vicreg_cov_loss": cov_loss.item(),
            "vicreg_std_pred": std_pred.mean().item(),
            "vicreg_std_target": std_target.mean().item(),
        }

        return total_vicreg, metrics


class NoemaWorldModelLoss(nn.Module):
    """
    Unified Loss for Noema-JEPA v1:
    L_total = lambda_pred * L_pred
            + lambda_vicreg * L_vicreg
            + lambda_intra * L_intra_var
            + lambda_think * L_think_aux
    """

    def __init__(self, config: Optional[NoemaConfig] = None) -> None:
        super().__init__()
        self.config = config or NoemaConfig()
        cfg = self.config

        self.vicreg = VICRegLoss(
            sim_coeff=cfg.sim_loss_weight,
            var_coeff=cfg.var_loss_weight,
            cov_coeff=cfg.cov_loss_weight,
            gamma=cfg.gamma_target,
        )

        self.pred_loss_weight = cfg.pred_loss_weight
        self.intra_var_loss_weight = cfg.intra_var_loss_weight
        self.min_intra_std = cfg.min_intra_std
        self.think_loss_weight = cfg.think_loss_weight

    def update_think_loss_weight(self, new_weight: float) -> None:
        """Allows VarianceWatchdog to dynamically adapt ThinkJEPA influence."""
        self.think_loss_weight = float(new_weight)

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates all loss components from NoemaJEPA forward pass outputs.

        Args:
            outputs: Dictionary returned by NoemaJEPA.forward().

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        pred_masked = outputs["pred_masked_slots"]      # [B, M, D]
        target_masked = outputs["target_masked_slots"]  # [B, M, D]
        all_pred_slots = outputs["all_pred_slots"]      # [B, S, D]
        pred_vicreg = outputs["pred_vicreg_proj"]       # [B*M, proj_dim]
        target_vicreg = outputs["target_vicreg_proj"]   # [B*M, proj_dim]
        pred_think_proj = outputs["pred_think_proj"]   # [B, D]
        c_scene = outputs["c_scene"]                   # [B, D]

        # 1. Masked slot reconstruction prediction loss (Smooth L1)
        pred_loss = F.smooth_l1_loss(pred_masked, target_masked.detach())

        # 2. VICReg loss
        vicreg_total, vicreg_metrics = self.vicreg(pred_vicreg, target_vicreg)

        # 3. Intra-scene variance penalty
        # Protects against FiLM flattening all object slots within a scene to the same representation
        # Computed in fp32
        slots_f32 = all_pred_slots.float()  # [B, S, D]
        # Variance across slots dimension for each sample in batch: [B, D]
        # unbiased=False prevents NaN if num_slots is 1
        intra_var = slots_f32.var(dim=1, unbiased=False) + 1e-4
        intra_std = torch.sqrt(intra_var).mean(dim=-1)  # [B]
        mean_intra_std = intra_std.mean()
        intra_scene_loss = torch.mean(F.relu(self.min_intra_std - intra_std))

        # 4. ThinkJEPA semantic alignment auxiliary loss (Cosine similarity)
        # pooled slot representation aligned with c_scene
        cos_sim = F.cosine_similarity(pred_think_proj, c_scene.detach(), dim=-1)
        think_aux_loss = torch.mean(1.0 - cos_sim)

        # 5. Total composite loss
        total_loss = (
            self.pred_loss_weight * pred_loss
            + vicreg_total
            + self.intra_var_loss_weight * intra_scene_loss
            + self.think_loss_weight * think_aux_loss
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss": total_loss.item(),
            "loss_pred": pred_loss.item(),
            "pred_loss": pred_loss.item(),
            "loss_vicreg": vicreg_total.item(),
            "vicreg_loss": vicreg_total.item(),
            "loss_intra_scene": intra_scene_loss.item(),
            "loss_think_aux": think_aux_loss.item(),
            "metric_intra_scene_std": mean_intra_std.item(),
            "intra_std": mean_intra_std.item(),
            "active_think_loss_weight": self.think_loss_weight,
            **vicreg_metrics,
        }

        return total_loss, metrics

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Alias for forward pass to satisfy explicit compute_loss calls."""
        return self.forward(outputs)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    config: Optional[NoemaConfig] = None,
    loss_fn: Optional[NoemaWorldModelLoss] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standalone functional interface for computing Noema-JEPA world model loss.
    Compatible with both 4D [B, C, H, W] images and 5D [B, T, C, H, W] video clips.
    """
    if loss_fn is None:
        cfg = config or NoemaConfig()
        device = outputs["pred_masked_slots"].device
        loss_fn = NoemaWorldModelLoss(cfg).to(device)
    return loss_fn(outputs)

