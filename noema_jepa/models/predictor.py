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
        action_dim: int = 16,
        film_alpha_init: float = 0.05,
        cond_dropout: float = 0.15,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_slots = max_slots
        self.action_dim = action_dim

        # Learnable MASK token for masked object slots
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learnable slot positional identity embeddings
        self.slot_pos_embed = nn.Parameter(torch.randn(1, max_slots, embed_dim) * 0.02)

        # Action conditioning projection for interactive World Model dynamics
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.action_proj[-1].weight)
        nn.init.zeros_(self.action_proj[-1].bias)

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
        action: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: Clean target object slots [B, num_slots, embed_dim].
            mask: Boolean mask [B, num_slots] where True indicates slots to be MASKED and predicted.
            c_scene: ThinkJEPA global semantic context vector [B, embed_dim].
            action: Optional action vector [B, action_dim].
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

        # Inject action conditioning if provided
        if action is not None:
            x = x + self.action_proj(action).unsqueeze(1)

        # Process through modulated relational blocks
        for blk in self.blocks:
            x = blk(x, c_scene=c_scene, modulation_scale=modulation_scale)

        x = self.norm(x)
        predicted_slots = self.pred_head(x)

        # Extract predictions specifically at masked positions from predicted_slots
        # (Fixes v1 bug where pred_head was erroneously executed twice)
        num_masked = mask[0].sum().item()
        pred_masked = predicted_slots[mask].reshape(B, num_masked, D)

        return pred_masked, predicted_slots

    def predict_next_step(
        self,
        slots: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        c_scene: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predicts next-step object slots S_{t+1} from current slots S_t and optional action a_t.

        Args:
            slots: Object slots at time t of shape [B, num_slots, embed_dim].
            action: Action vector [B, action_dim] or None.
            c_scene: Semantic context [B, embed_dim] or None.
            modulation_scale: FiLM scale multiplier.

        Returns:
            predicted_next_slots: [B, num_slots, embed_dim].
        """
        B, S, D = slots.shape
        pos = self.slot_pos_embed[:, :S, :]
        x = slots + pos

        if action is not None:
            x = x + self.action_proj(action).unsqueeze(1)

        for blk in self.blocks:
            x = blk(x, c_scene=c_scene, modulation_scale=modulation_scale)

        x = self.norm(x)
        return self.pred_head(x)

    def forward_temporal(
        self,
        slots_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        c_scene: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Rollout temporal slot transitions across frames S_{0:T-1} -> S_hat_{1:T}.

        Args:
            slots_seq: Sequence of slot states [B, T, num_slots, embed_dim].
            actions: Optional action tensor [B, T, action_dim] or [B, action_dim].
            c_scene: Semantic context [B, embed_dim] or None.
            modulation_scale: FiLM scale multiplier.

        Returns:
            pred_future_slots: Predicted next-step slots [B, T-1, num_slots, embed_dim].
        """
        B, T, S, D = slots_seq.shape
        pred_steps = []
        for t in range(T - 1):
            act_t = None
            if actions is not None:
                act_t = actions[:, t] if actions.ndim == 3 else actions
            pred_t = self.predict_next_step(
                slots=slots_seq[:, t],
                action=act_t,
                c_scene=c_scene,
                modulation_scale=modulation_scale,
            )
            pred_steps.append(pred_t)
        return torch.stack(pred_steps, dim=1)
