"""
Spatial Patch Encoder and Slot Attention Object Discovery Module.

Supports:
- Static images [B, C, H, W] or video clips [B, T, C, H, W].
- Numerically stable Slot Attention (AMP fp16/bf16 safe) for discovering discrete object slots.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpatialPatchEncoder(nn.Module):
    """
    Vision encoder mapping images [B, C, H, W] or short videos [B, T, C, H, W]
    into contextual spatial patch representations [B, N_patches, D].
    """

    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 288,
        depth: int = 5,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 8,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Patch embedding projection
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable spatial 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Learnable temporal embeddings for video support
        self.temporal_embed = nn.Parameter(torch.randn(1, max_frames, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, return_per_frame: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor, either [B, C, H, W] or [B, T, C, H, W].
            return_per_frame: If True and input is 5D video, returns [B, T, N_patches, D].

        Returns:
            Spatial patch tokens of shape [B, N_patches, embed_dim] or [B, T, N_patches, embed_dim].
        """
        is_video = (x.ndim == 5)
        if is_video:
            B, T, C, H, W = x.shape
            # Flatten B and T to process through 2D patch embed
            x_flat = rearrange(x, "b t c h w -> (b t) c h w")
            patches = self.patch_embed(x_flat)  # [(B*T), D, H_grid, W_grid]
            patches = rearrange(patches, "(b t) d h w -> b t (h w) d", b=B, t=T)
            # Add spatial embeddings
            patches = patches + self.pos_embed.unsqueeze(1)
            
            # Temporal embeddings: slice if T <= max_frames, otherwise interpolate
            max_t = self.temporal_embed.size(1)
            if T <= max_t:
                temp_emb = self.temporal_embed[:, :T]
            else:
                temp_raw = self.temporal_embed.squeeze(2).permute(0, 2, 1)  # [1, D, max_t]
                temp_interp = F.interpolate(temp_raw, size=T, mode="linear", align_corners=False)
                temp_emb = temp_interp.permute(0, 2, 1).unsqueeze(2)  # [1, T, 1, D]

            patches = patches + temp_emb
            # Flatten spatial and temporal dimensions into a single sequence: [B, T * N_patches, D]
            tokens = rearrange(patches, "b t n d -> b (t n) d")
        else:
            B, C, H, W = x.shape
            patches = self.patch_embed(x)  # [B, D, H_grid, W_grid]
            patches = rearrange(patches, "b d h w -> b (h w) d")
            tokens = patches + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        if is_video and return_per_frame:
            return rearrange(tokens, "b (t n) d -> b t n d", t=T, n=self.num_patches)
        elif (not is_video) and return_per_frame:
            return tokens.unsqueeze(1)
        return tokens


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer Encoder block with FlashAttention (scaled_dot_product_attention)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # Attention with Pre-LN
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.proj(attn_out)

        # MLP with Pre-LN
        x = x + self.mlp(self.norm2(x))
        return x


class SlotAttention(nn.Module):
    """
    Slot Attention module for Object Discovery (Locatello et al., 2020).
    Maps continuous spatial patch representations into discrete object slots.
    
    Includes AMP-stabilized arithmetic:
    - Dot product scaling and softmax evaluated with float32 casting to eliminate fp16 underflows.
    - Robust epsilon additions for denominator sum normalization.
    """

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 288,
        in_features: int = 288,
        iters: int = 3,
        hidden_dim: int = 576,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Slot initialization parameters (learned Gaussian distribution)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)

        # Projections for spatial features and slots
        self.norm_inputs = nn.LayerNorm(in_features)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ffn = nn.LayerNorm(slot_dim)

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(in_features, slot_dim, bias=False)
        self.project_v = nn.Linear(in_features, slot_dim, bias=False)

        # Recurrent state transition and MLP refinement
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # SAVi-style temporal transition predictor for recurrent tracking across frames
        self.temporal_transition = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )
        # Zero-initialize the final projection so transition starts as exact identity
        nn.init.zeros_(self.temporal_transition[-1].weight)
        nn.init.zeros_(self.temporal_transition[-1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: Optional[int] = None,
        init_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Spatial feature tokens [B, N_patches, in_features].
            num_slots: Optional override for slot count.
            init_slots: Optional initial slot representations [B, K, slot_dim].
                        If provided, bypasses learned Gaussian sampling (SAVi tracking).

        Returns:
            Discovered object slots of shape [B, num_slots, slot_dim].
        """
        B, N, _ = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots

        # Initialize slot representations
        if init_slots is not None:
            slots = init_slots
        else:
            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_log_sigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn_like(mu)

        # Project inputs to keys and values once
        inputs_norm = self.norm_inputs(inputs)
        k = self.project_k(inputs_norm)  # [B, N, D]
        v = self.project_v(inputs_norm)  # [B, N, D]

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)  # [B, K, D]

            # Numerically stable attention: compute dots in float32
            # Shape: [B, K, N]
            dots = torch.einsum("bkd,bnd->bkn", q.float(), k.float()) * self.scale
            
            # Softmax over SLOTS dimension (slots compete for patches)
            attn = dots.softmax(dim=1)  # [B, K, N]

            # Re-normalize along patch dimension (with eps safe guard)
            attn_norm = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn_norm = attn_norm.to(v.dtype)

            # Weighted patch aggregation for each slot
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)  # [B, K, D]

            # GRU update
            slots_flat = slots_prev.reshape(B * K, self.slot_dim)
            updates_flat = updates.reshape(B * K, self.slot_dim)
            slots_updated = self.gru(updates_flat, slots_flat)
            slots = slots_updated.reshape(B, K, self.slot_dim)

            # Residual MLP refinement
            slots = slots + self.mlp(self.norm_pre_ffn(slots))

        return slots

    def forward_video(
        self,
        frame_features: torch.Tensor,
        num_slots: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SAVi-style recurrent slot propagation across a temporal sequence of frames.

        Args:
            frame_features: Spatial features per frame of shape [B, T, N, in_features].
            num_slots: Optional override for slot count.

        Returns:
            trajectories: Slot states across all time steps [B, T, K, slot_dim].
            consolidated: Consolidated slot representations at the final frame [B, K, slot_dim].
        """
        B, T, N, _ = frame_features.shape
        K = num_slots if num_slots is not None else self.num_slots

        trajectories = []
        current_slots: Optional[torch.Tensor] = None

        for t in range(T):
            feat_t = frame_features[:, t]  # [B, N, in_features]
            if t == 0:
                current_slots = self.forward(feat_t, num_slots=K, init_slots=None)
            else:
                pred_init = current_slots + self.temporal_transition(current_slots)
                current_slots = self.forward(feat_t, num_slots=K, init_slots=pred_init)
            trajectories.append(current_slots)

        trajectories_tensor = torch.stack(trajectories, dim=1)  # [B, T, K, slot_dim]
        consolidated = trajectories_tensor[:, -1]               # [B, K, slot_dim]
        return trajectories_tensor, consolidated
