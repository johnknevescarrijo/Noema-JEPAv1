import copy
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from noema_jepa.config import NoemaConfig
from noema_jepa.models.encoder import SpatialPatchEncoder, SlotAttention
from noema_jepa.models.film import ZeroInitResidualFiLM
from noema_jepa.models.quantizer import FSQSlotQuantizer
from noema_jepa.models.predictor import CJEPAObjectPredictor


class ThinkJEPABridge(nn.Module):
    """
    Projects semantic reasoning embeddings from a frozen VLM (e.g. Qwen2-VL, InternVL2)
    into the Noema-JEPA latent space.
    """

    def __init__(self, vlm_dim: int = 1536, embed_dim: int = 288) -> None:
        super().__init__()
        self.vlm_dim = vlm_dim
        self.embed_dim = embed_dim

        # Learnable fallback context vector for ablation or pure vision inference
        self.fallback_context = nn.Parameter(torch.zeros(1, vlm_dim))

        self.projector = nn.Sequential(
            nn.Linear(vlm_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, think_embeddings: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Args:
            think_embeddings: Optional [B, vlm_dim] or [B, T, vlm_dim] semantic context vectors from frozen VLM.
            batch_size: Used if think_embeddings is None to generate matching batch dimension.

        Returns:
            Projected scene context vector c_scene of shape [B, embed_dim].
        """
        if (
            think_embeddings is None
            or not isinstance(think_embeddings, torch.Tensor)
            or think_embeddings.numel() == 0
        ):
            emb = self.fallback_context.expand(batch_size, -1)
        else:
            emb = think_embeddings
            # Handle 1D tensor [vlm_dim] -> [1, vlm_dim]
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            # Handle 3D tensor [B, T, vlm_dim] -> pool over temporal dimension to [B, vlm_dim]
            elif emb.ndim == 3:
                emb = emb.mean(dim=1)
            # Handle flattened temporal batch [B*T, vlm_dim] when batch_size B is provided
            elif emb.ndim == 2 and batch_size > 0 and emb.size(0) != batch_size and (emb.size(0) % batch_size == 0):
                emb = emb.view(batch_size, -1, self.vlm_dim).mean(dim=1)

        return self.projector(emb)


class NoemaJEPA(nn.Module):
    """
    Noema-JEPA World Model (v2):
    1. Visual Encoder (Student & optional EMA Target): Spatial patches from image/video.
    2. Slot Attention (SAVi recurrent tracking across frames): Discovers object slots.
    3. FSQ Symbolic Quantizer: Finite Scalar Quantization bottleneck for object tokens.
    4. ThinkJEPA Bridge: Projects global reasoning context from frozen VLM.
    5. C-JEPA Predictor: Masked slot prediction + multi-step temporal rollout with action conditioning.
    6. Projections: LayerNorm-stabilized VICReg projector + ThinkJEPA semantic alignment head.
    """

    def __init__(self, config: Optional[NoemaConfig] = None) -> None:
        super().__init__()
        self.config = config or NoemaConfig()
        cfg = self.config

        # 1. Spatial Patch Encoder (Student)
        self.encoder = SpatialPatchEncoder(
            img_size=cfg.img_size,
            in_channels=cfg.in_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        # 2. Slot Attention (Student)
        self.slot_attention = SlotAttention(
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            in_features=cfg.embed_dim,
            iters=cfg.slot_iters,
            hidden_dim=cfg.slot_hidden_dim,
        )

        # 3. EMA Target Encoder (v2: frozen teacher for stable latent targets)
        if getattr(cfg, "use_ema_target", True):
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_slot_attention = copy.deepcopy(self.slot_attention)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            for p in self.target_slot_attention.parameters():
                p.requires_grad = False

        # 4. FSQ Symbolic Slot Quantizer (v2)
        if getattr(cfg, "use_fsq_quantizer", True):
            self.fsq_quantizer = FSQSlotQuantizer(
                dim=cfg.slot_dim,
                levels=getattr(cfg, "fsq_levels", (8, 5, 5, 5)),
                residual_weight=getattr(cfg, "fsq_residual_weight", 0.15),
            )

        # 5. ThinkJEPA Bridge
        self.think_bridge = ThinkJEPABridge(
            vlm_dim=cfg.vlm_dim,
            embed_dim=cfg.embed_dim,
        )

        # 6. Contextual JEPA Relational Predictor with FiLM and Action Conditioning
        self.predictor = CJEPAObjectPredictor(
            embed_dim=cfg.embed_dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_heads,
            mlp_ratio=cfg.mlp_ratio,
            max_slots=cfg.num_slots,
            action_dim=getattr(cfg, "action_dim", 16),
            film_alpha_init=cfg.film_alpha_init,
            cond_dropout=cfg.cond_dropout,
            dropout=cfg.dropout,
        )

        # 7. VICReg Projector (v2: LayerNorm replaces BatchNorm1d to eliminate variance shrinkage)
        if getattr(cfg, "vicreg_use_layernorm", True):
            self.vicreg_projector = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.vicreg_proj_dim),
                nn.LayerNorm(cfg.vicreg_proj_dim),
                nn.GELU(),
                nn.Linear(cfg.vicreg_proj_dim, cfg.vicreg_proj_dim),
            )
        else:
            self.vicreg_projector = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.vicreg_proj_dim),
                nn.BatchNorm1d(cfg.vicreg_proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.vicreg_proj_dim, cfg.vicreg_proj_dim),
            )

        # 8. ThinkJEPA Semantic Alignment Projector
        self.think_align_projector = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

    @torch.no_grad()
    def update_ema(self, momentum: Optional[float] = None) -> None:
        """
        Exponential Moving Average update of target encoder and target slot attention:
        param_target = momentum * param_target + (1 - momentum) * param_student
        """
        if not getattr(self.config, "use_ema_target", True):
            return
        if not hasattr(self, "target_encoder") or not hasattr(self, "target_slot_attention"):
            return

        m = momentum if momentum is not None else getattr(self.config, "ema_decay_base", 0.996)

        for p_s, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

        for p_s, p_t in zip(self.slot_attention.parameters(), self.target_slot_attention.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    def generate_mask(
        self,
        batch_size: int,
        num_slots: int,
        num_masked: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generates a boolean mask [B, num_slots] with exactly `num_masked` True values per row.
        True denotes an occluded/masked object slot to be predicted.
        """
        assert num_masked < num_slots, f"num_masked ({num_masked}) must be < num_slots ({num_slots})"
        rand_indices = torch.rand(batch_size, num_slots, device=device).argsort(dim=1)
        mask = rand_indices < num_masked
        return mask

    def forward(
        self,
        images: torch.Tensor,
        think_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        modulation_scale: float = 1.0,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of Noema-JEPA v2.

        Args:
            images: Input images [B, C, H, W] or video clips [B, T, C, H, W].
            think_embeddings: Optional [B, vlm_dim] frozen VLM guidance vectors.
            mask: Optional boolean mask [B, num_slots] for masked object slots.
            modulation_scale: Dynamic FiLM scaling factor from VarianceWatchdog.
            actions: Optional action conditioning [B, action_dim] or [B, T, action_dim].

        Returns:
            Dictionary containing predicted slots, targets, projected latents, and diagnostics.
        """
        # Support flexible kwargs (e.g. action alias)
        if actions is None and "action" in kwargs:
            actions = kwargs["action"]

        B = images.size(0)
        device = images.device
        is_video = (images.ndim == 5)
        use_temporal = is_video and getattr(self.config, "use_temporal_slot_tracking", True)
        use_ema = getattr(self.config, "use_ema_target", True) and hasattr(self, "target_encoder")

        # 1. Feature extraction & slot discovery (SAVi video mode vs static image mode)
        if use_temporal:
            # Student pass
            student_features = self.encoder(images, return_per_frame=True)  # [B, T, N, D]
            student_trajectories, student_slots = self.slot_attention.forward_video(student_features)

            # Target pass (Teacher EMA or detached student)
            if use_ema:
                with torch.no_grad():
                    target_features = self.target_encoder(images, return_per_frame=True)
                    target_trajectories, target_slots = self.target_slot_attention.forward_video(target_features)
            else:
                target_trajectories = student_trajectories.detach()
                target_slots = student_slots.detach()
        else:
            # Student pass
            student_features = self.encoder(images)  # [B, N, D]
            student_slots = self.slot_attention(student_features)  # [B, K, D]

            # Target pass
            if use_ema:
                with torch.no_grad():
                    target_features = self.target_encoder(images)
                    target_slots = self.target_slot_attention(target_features)
            else:
                target_slots = student_slots.detach()

        # 2. FSQ Symbolic Quantization on Student slots
        fsq_metrics = {}
        if getattr(self.config, "use_fsq_quantizer", True) and hasattr(self, "fsq_quantizer"):
            context_slots, fsq_metrics = self.fsq_quantizer(student_slots)
        else:
            context_slots = student_slots

        # 3. ThinkJEPA semantic context projection
        c_scene = self.think_bridge(think_embeddings, batch_size=B)  # [B, D]

        # 4. Generate C-JEPA Object Mask if not provided
        if mask is None:
            mask = self.generate_mask(
                batch_size=B,
                num_slots=self.config.num_slots,
                num_masked=self.config.num_masked_slots,
                device=device,
            )

        # 5. Extract ground-truth target masked slots
        num_masked = self.config.num_masked_slots
        target_masked_slots = target_slots[mask].reshape(B, num_masked, self.config.embed_dim)

        # 6. Predict masked object representations conditioned on visible objects, c_scene, and actions
        pred_action = actions[:, -1] if (actions is not None and actions.ndim == 3) else actions
        pred_masked_slots, all_pred_slots = self.predictor(
            slots=context_slots,
            mask=mask,
            c_scene=c_scene,
            action=pred_action,
            modulation_scale=modulation_scale,
        )

        # 7. VICReg projections (computed over masked slots)
        pred_flat = pred_masked_slots.reshape(-1, self.config.embed_dim)
        target_flat = target_masked_slots.detach().reshape(-1, self.config.embed_dim)

        pred_vicreg_proj = self.vicreg_projector(pred_flat)
        target_vicreg_proj = self.vicreg_projector(target_flat)

        # 8. ThinkJEPA semantic alignment projection (mean-pooled across all object slots)
        pooled_pred_slots = all_pred_slots.mean(dim=1)  # [B, D]
        pred_think_proj = self.think_align_projector(pooled_pred_slots)  # [B, D]

        outputs = {
            "pred_masked_slots": pred_masked_slots,
            "target_masked_slots": target_masked_slots,
            "all_pred_slots": all_pred_slots,
            "target_slots": target_slots,
            "context_slots": context_slots,
            "pred_vicreg_proj": pred_vicreg_proj,
            "target_vicreg_proj": target_vicreg_proj,
            "pred_think_proj": pred_think_proj,
            "c_scene": c_scene,
            "mask": mask,
            "modulation_scale": torch.tensor(modulation_scale, device=device),
            **fsq_metrics,
        }

        # 9. Multi-step temporal rollout prediction when in video mode with T > 1
        if use_temporal and images.size(1) > 1:
            T = images.size(1)
            pred_next_slots = self.predictor.forward_temporal(
                slots_seq=student_trajectories,
                actions=actions,
                c_scene=c_scene,
                modulation_scale=modulation_scale,
            )
            target_next_slots = target_trajectories[:, 1:].detach()

            outputs["pred_next_slots"] = pred_next_slots
            outputs["target_next_slots"] = target_next_slots
            outputs["slot_trajectories"] = student_trajectories
            outputs["target_slot_trajectories"] = target_trajectories

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Convenience method to compute multi-component loss directly from model outputs.
        Ensures compatibility with both 4D [B, C, H, W] and 5D [B, T, C, H, W] outputs.
        """
        if loss_fn is None:
            if not hasattr(self, "_loss_fn_cache"):
                from noema_jepa.losses.vicreg import NoemaWorldModelLoss
                device = outputs["pred_masked_slots"].device
                self._loss_fn_cache = NoemaWorldModelLoss(self.config).to(device)
            loss_fn = self._loss_fn_cache
        return loss_fn(outputs)

    def parameter_breakdown(self, print_table: bool = True) -> Dict[str, int]:
        """
        Computes exact parameter counts for all submodules.
        Prints a neatly formatted table verifying the ~15M parameter target.
        """
        modules = {
            "SpatialPatchEncoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ThinkJEPABridge": self.think_bridge,
            "CJEPAObjectPredictor": self.predictor,
            "VICRegProjector": self.vicreg_projector,
            "ThinkAlignProjector": self.think_align_projector,
        }
        if hasattr(self, "fsq_quantizer"):
            modules["FSQSlotQuantizer"] = self.fsq_quantizer

        counts = {}
        total = 0
        for name, mod in modules.items():
            cnt = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            counts[name] = cnt
            total += cnt

        counts["Total"] = total
        counts["Total Trainable Parameters"] = total

        if print_table:
            v_tag = getattr(self.config, "model_version", "v2").upper()
            title = f"Noema-JEPA {v_tag} Parameter Breakdown"
            print("\n" + "=" * 65)
            print(f"{title:^65}")
            print("=" * 65)
            print(f"{'Module':<25} | {'Parameters':>15} | {'Percentage':>12}")
            print("-" * 65)
            for name, cnt in counts.items():
                if name == "Total":
                    print("-" * 65)
                    pct = 100.0
                else:
                    pct = (cnt / total) * 100.0 if total > 0 else 0.0
                print(f"{name:<25} | {cnt:>15,} | {pct:>11.2f}%")
            print("=" * 65)
            print(f"Target: ~15.0M | Actual: {total / 1e6:.3f}M trainable parameters\n")

        return counts
