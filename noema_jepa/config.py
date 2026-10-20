"""
Configuration dataclass for Noema-JEPA v1 architecture (~15M parameters).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class NoemaConfig:
    """
    Master configuration for Noema-JEPA v1.
    
    Default dimensions are calibrated to yield approximately 15.0M trainable parameters:
    - embed_dim: 288
    - encoder_depth: 5
    - predictor_depth: 6
    - slot_dim: 288
    - num_slots: 8
    - vlm_dim: 1536
    - vicreg_proj_dim: 1024
    """

    # --- Input Dimensions ---
    img_size: int = 128
    in_channels: int = 3
    num_frames: int = 1  # 1 for static images [B, C, H, W], >1 for short video clips [B, T, C, H, W]
    patch_size: int = 16

    # --- Core Model Dimensions (~15M target) ---
    embed_dim: int = 288
    encoder_depth: int = 5
    encoder_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # --- Slot Attention (Object Discovery) ---
    num_slots: int = 8
    slot_dim: int = 288
    slot_iters: int = 3
    slot_hidden_dim: int = 576  # MLP expansion in slot attention (2x slot_dim)

    # --- Predictor (Contextual JEPA / Relational Transformer) ---
    predictor_depth: int = 6
    predictor_heads: int = 6
    num_masked_slots: int = 4  # Number of slots masked out per scene during C-JEPA training

    # --- ThinkJEPA Bridge & FiLM Modulation ---
    vlm_dim: int = 1536  # Dimensionality of the frozen VLM embeddings (e.g. Qwen2-VL, InternVL2)
    film_alpha_init: float = 0.05  # Small initial gate for stable cold-start
    cond_dropout: float = 0.15  # FiLM conditional dropout (Classifier-Free Guidance style)
    initial_film_modulation_scale: float = 1.0

    # --- Projections & Latent Spaces ---
    vicreg_proj_dim: int = 768  # Projector latent dim for VICReg loss (calibrated for 15.0M total params)
    vicreg_proj_depth: int = 2   # Number of MLP layers in VICReg projector

    # --- Loss Weights & Hyperparameters ---
    pred_loss_weight: float = 1.0       # Weight for masked slot prediction loss
    sim_loss_weight: float = 1.0        # VICReg Invariance loss weight (s)
    var_loss_weight: float = 1.0        # VICReg Batch Variance loss weight (v)
    cov_loss_weight: float = 0.04       # VICReg Covariance loss weight (c)
    intra_var_loss_weight: float = 0.5  # Intra-scene variance penalty weight
    min_intra_std: float = 0.1          # Minimum required std among objects within the same scene
    gamma_target: float = 1.0           # Desired standard deviation target for VICReg batch std
    think_loss_weight: float = 0.1      # ThinkJEPA alignment loss weight
    think_temperature: float = 0.07     # Temperature for semantic alignment (Cosine / InfoNCE)

    # --- Variance Watchdog Hyperparameters ---
    watchdog_std_threshold: float = 0.05    # Critical threshold for batch and intra-scene std
    watchdog_recovery_factor: float = 1.5   # Multiplier above threshold to consider representation healthy
    watchdog_decay_rate: float = 0.8        # Multiplicative damping factor when collapse warning fires
    watchdog_recovery_rate: float = 1.05    # Multiplicative recovery factor towards nominal scale
    min_film_modulation_scale: float = 0.05 # Minimum clamp for FiLM scale
    max_film_modulation_scale: float = 1.0  # Maximum clamp for FiLM scale

    # --- vLLM Think Extractor / Colab Cache ---
    vllm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_dtype: str = "half"  # "half", "bfloat16", or "float16"
    vllm_max_model_len: int = 2048
    cache_dir: str = "./cache/think_embeddings"

    # --- Training & Colab Optimization ---
    batch_size: int = 32
    learning_rate: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" for Colab T4, "bfloat16" for L4 / A100
    grad_clip_norm: float = 1.0
    use_compile: bool = False   # Optional torch.compile (requires Triton)

    def validate(self) -> None:
        """Sanity check configuration integrity."""
        assert self.embed_dim % self.encoder_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by encoder_heads ({self.encoder_heads})"
        )
        assert self.embed_dim % self.predictor_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by predictor_heads ({self.predictor_heads})"
        )
        assert self.num_masked_slots < self.num_slots, (
            f"num_masked_slots ({self.num_masked_slots}) must be less than total num_slots ({self.num_slots})"
        )
