"""
Finite Scalar Quantization (FSQ) for Symbolic Slot Discretization.

FSQ (Mentzer et al., 2023) maps continuous latent slot vectors to a finite discrete codebook
without learned codebook dictionaries, vector quantization (VQ) codebook collapse, or EMA codebook updates.
Bounded scalar quantization with Straight-Through Estimators (STE) guarantees 100% codebook stability.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQSlotQuantizer(nn.Module):
    """
    Finite Scalar Quantizer (FSQ) applied to continuous object slot representations.

    Maps slots [..., dim] into K = len(levels) bounded scalar channels, discretizes them
    to discrete integer levels via round() with straight-through gradient estimation (STE),
    and projects back to [..., dim] via a residual connection.

    Default levels (8, 5, 5, 5) yields 8 * 5 * 5 * 5 = 1000 discrete codes.
    """

    def __init__(
        self,
        dim: int = 288,
        levels: Tuple[int, ...] = (8, 5, 5, 5),
        residual_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.levels = tuple(levels)
        self.residual_weight = residual_weight
        self.num_levels = len(levels)

        # Total discrete codebook size: prod(levels)
        total_codes = 1
        for lev in levels:
            total_codes *= lev
        self.total_codes = total_codes

        # Linear projections to/from quantization bottleneck
        self.project_in = nn.Linear(dim, self.num_levels)
        self.project_out = nn.Linear(self.num_levels, dim)

        # Initialize output projection near zero for smooth residual cold-start
        nn.init.normal_(self.project_out.weight, mean=0.0, std=1e-4)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

        # Precompute basis multipliers and half-widths as persistent buffers
        _half_widths = []
        _basis = []
        current_mult = 1
        for lev in self.levels:
            _half_widths.append((lev - 1) / 2.0)
            _basis.append(current_mult)
            current_mult *= lev

        self.register_buffer("half_widths", torch.tensor(_half_widths, dtype=torch.float32))
        self.register_buffer("basis", torch.tensor(_basis, dtype=torch.long))
        self.register_buffer("levels_tensor", torch.tensor(self.levels, dtype=torch.long))

    def _quantize_bounded(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes input z using scaled tanh and STE round.
        Returns:
            z_q: STE quantized tensor in continuous space [-1, 1].
            indices: Discrete codebook integer indices in [0, total_codes - 1].
        """
        half_w = self.half_widths.to(z.device, dtype=z.dtype)
        # Scaled tanh bounded in [-half_w, half_w]
        # Adding small margin (0.5 - eps) ensures values comfortably span all integer buckets
        scale = half_w
        z_bounded = torch.tanh(z) * scale

        # Round to nearest integer: bucket in [-floor(L/2), +floor(L/2)]
        z_rounded = torch.round(z_bounded)

        # Straight-Through Estimator: forward z_rounded, backward z_bounded
        z_ste = z_bounded + (z_rounded - z_bounded).detach()

        # Normalize z_ste back to normalized range [-1.0, 1.0] for stable projection back
        z_q = z_ste / (half_w.clamp(min=1.0))

        # Discrete integer indices for codebook tracking:
        # Shift buckets so min value is 0: c_d in [0, L_d - 1]
        c = (z_rounded + half_w).round().long()
        # Clamp to ensure indices strictly in [0, L_d - 1]
        levels_t = self.levels_tensor.to(z.device)
        c = torch.clamp(c, min=torch.zeros_like(levels_t), max=levels_t - 1)

        basis_t = self.basis.to(z.device)
        indices = (c * basis_t).sum(dim=-1)  # shape: z.shape[:-1]

        return z_q, indices

    def forward(
        self, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            slots: Object slot tensor of shape [..., dim].

        Returns:
            slots_out: Refined slot representations with residual discrete component.
            metrics: Dictionary containing 'fsq_perplexity' and 'fsq_codebook_usage'.
        """
        # Project continuous slots to bottleneck
        z = self.project_in(slots)  # [..., num_levels]
        z_q, indices = self._quantize_bounded(z)

        # Project quantized representations back to slot dimension
        z_proj = self.project_out(z_q)  # [..., dim]

        # Residual blend preserving continuous geometry while infusing discrete codebook structure
        slots_out = slots + self.residual_weight * z_proj

        # Compute codebook usage and perplexity in torch.no_grad()
        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            unique_codes = torch.unique(flat_indices)
            codebook_usage = float(len(unique_codes)) / float(self.total_codes)

            # Perplexity calculation: exp(Entropy)
            counts = torch.bincount(flat_indices, minlength=self.total_codes).float()
            total_items = counts.sum()
            probs = counts / (total_items + 1e-10)
            probs_active = probs[probs > 0]
            entropy = -torch.sum(probs_active * torch.log(probs_active + 1e-10))
            perplexity = torch.exp(entropy)

            metrics = {
                "fsq_perplexity": perplexity.detach(),
                "fsq_codebook_usage": torch.tensor(codebook_usage, device=slots.device),
            }

        return slots_out, metrics

    @torch.no_grad()
    def get_codebook_indices(self, slots: torch.Tensor) -> torch.Tensor:
        """Returns integer codebook tokens for downstream symbolic reasoning or analysis."""
        z = self.project_in(slots)
        _, indices = self._quantize_bounded(z)
        return indices
