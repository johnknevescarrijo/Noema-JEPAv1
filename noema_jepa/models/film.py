"""
Zero-Initialized Residual FiLM (Feature-wise Linear Modulation) layer.

Provides smooth, non-disruptive semantic conditioning from the ThinkJEPA VLM context
into the C-JEPA relational object predictor tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ZeroInitResidualFiLM(nn.Module):
    """
    Zero-Initialized Residual FiLM Layer:
        h_norm = LayerNorm(h)
        eff_alpha = alpha * modulation_scale
        h_out = (1.0 + eff_alpha * tanh(gamma)) * h_norm + (eff_alpha * beta)

    Key Features:
    1. Learnable gate `alpha` initialized to a small value (e.g. 0.05) ensuring cold-start stability.
    2. Zero/near-zero initialization for linear projection weights to prevent early gradient shock.
    3. External dynamic `modulation_scale` governed by VarianceWatchdog to counter representation collapse.
    4. Conditional Dropout (Classifier-Free Guidance style) during training.
    """

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        alpha_init: float = 0.05,
        cond_dropout: float = 0.15,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.cond_dropout = cond_dropout
        self.eps = eps

        # Layer normalization for incoming hidden states
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

        # Projection from conditioning vector c_scene into gamma and beta
        self.cond_proj = nn.Linear(cond_dim, 2 * embed_dim)

        # Learnable gating parameter alpha: initialized small to guarantee near-identity at start
        self.alpha = nn.Parameter(torch.full((1, 1, embed_dim), fill_value=alpha_init))

        # Initialize projection weights to zero/small Gaussian
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        c_scene: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            h: Hidden representation [B, N, D] (e.g. object slots).
            c_scene: Scene context vector [B, cond_dim] from ThinkJEPA Bridge.
                     If None, FiLM acts as identity + LayerNorm.
            modulation_scale: External scalar multiplier from VarianceWatchdog (0.0 to 1.0).

        Returns:
            Modulated hidden states [B, N, D].
        """
        h_norm = self.norm(h)

        if c_scene is None:
            return h_norm

        # Classifier-Free Guidance style conditioning dropout during training
        if self.training and self.cond_dropout > 0.0:
            # Drop entire context vector for random samples in batch
            drop_mask = torch.bernoulli(
                torch.full((c_scene.size(0), 1), 1.0 - self.cond_dropout, device=c_scene.device)
            )
            c_scene = c_scene * drop_mask

        # Project context vector to gamma and beta
        # c_scene is [B, cond_dim] -> proj is [B, 2*D] -> split into [B, 1, D] each
        gamma_beta = self.cond_proj(c_scene).unsqueeze(1)  # [B, 1, 2*D]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)   # [B, 1, D], [B, 1, D]

        # Effective modulation alpha, scaled dynamically by watchdog
        eff_alpha = self.alpha * float(modulation_scale)

        # Non-linear gating with tanh to prevent runaway scale
        scale = 1.0 + eff_alpha * torch.tanh(gamma)
        shift = eff_alpha * beta

        return scale * h_norm + shift

    def get_effective_alpha_norm(self, modulation_scale: float = 1.0) -> float:
        """Helper to inspect the active FiLM gate norm."""
        eff_alpha = (self.alpha * float(modulation_scale)).detach()
        return eff_alpha.abs().mean().item()
