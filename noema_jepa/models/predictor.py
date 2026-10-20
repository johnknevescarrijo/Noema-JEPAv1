"""
Contextual JEPA Object Predictor (C-JEPA Relational Transformer).

Processes visible object slots alongside learnable [MASK] tokens for occluded objects,
modulated at every relational layer by ThinkJEPA semantic vectors via ZeroInitResidualFiLM.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from noema_jepa.models.film import ZeroInitResidualFiLM


class CJEPABlock(nn.Module):
    """
    Relational Transformer Block with ZeroInitResidualFiLM modulation.
    
    Order of operations:
    1. Pre-LN Self-Attention (FlashAttention / scaled_dot_product_attention)
    2. ZeroInitResidualFiLM modulation conditioned on c_scene
    3. Pre-LN FeedForward (MLP)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: Optional[int] = None,
        film_alpha_init: float = 0.05,
        cond_dropout: float = 0.15,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        cond_dim = cond_dim or embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        # FiLM semantic modulation layer
        self.film = ZeroInitResidualFiLM(
            embed_dim=embed_dim,
            cond_dim=cond_dim,
            alpha_init=film_alpha_init,
            cond_dropout=cond_dropout,
        )

        # FeedForward block
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        c_scene: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> torch.Tensor:
        B, S, D = x.shape

        # 1. Self-Attention
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.proj(attn_out)

        # 2. FiLM Modulation
        x = self.film(x, c_scene=c_scene, modulation_scale=modulation_scale)

        # 3. FeedForward
        x = x + self.mlp(self.norm2(x))
        return x


class CJEPAObjectPredictor(nn.Module):
    """
    Predicts representations of masked object slots from visible object slots
    conditioned on ThinkJEPA semantic scene context.
    """

    def __init__(
        self,
        embed_dim: int = 288,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        max_slots: int = 16,
        film_alpha_init: float = 0.05,
        cond_dropout: float = 0.15,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_slots = max_slots

        # Learnable MASK token for masked object slots
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learnable slot positional identity embeddings
        self.slot_pos_embed = nn.Parameter(torch.randn(1, max_slots, embed_dim) * 0.02)

        # Stack of relational transformer blocks
        self.blocks = nn.ModuleList([
            CJEPABlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=embed_dim,
                film_alpha_init=film_alpha_init,
                cond_dropout=cond_dropout,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head mapping transformer hidden state to slot representation
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        mask: torch.Tensor,
        c_scene: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: Clean target object slots [B, num_slots, embed_dim].
            mask: Boolean mask [B, num_slots] where True indicates slots to be MASKED and predicted.
            c_scene: ThinkJEPA global semantic context vector [B, embed_dim].
            modulation_scale: Dynamic FiLM scale multiplier from VarianceWatchdog.

        Returns:
            Tuple of:
            - pred_masked: Predicted representations for the masked slots [B, num_masked, embed_dim].
            - full_predicted_slots: All slots after predictor refinement [B, num_slots, embed_dim].
        """
        B, S, D = slots.shape
        pos = self.slot_pos_embed[:, :S, :]

        # Replace masked slots with learnable mask token
        mask_expanded = mask.unsqueeze(-1)  # [B, S, 1]
        mask_token = self.mask_token.expand(B, S, -1)
        x = torch.where(mask_expanded, mask_token, slots)
        x = x + pos

        # Process through modulated relational blocks
        for blk in self.blocks:
            x = blk(x, c_scene=c_scene, modulation_scale=modulation_scale)

        x = self.norm(x)
        predicted_slots = self.pred_head(x)

        # Extract predictions specifically at masked positions
        # Using boolean indexing for masked slots per sample
        num_masked = mask[0].sum().item()
        pred_masked = x[mask].reshape(B, num_masked, D)
        pred_masked = self.pred_head(pred_masked)

        return pred_masked, predicted_slots
