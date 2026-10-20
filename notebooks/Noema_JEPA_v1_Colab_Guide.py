# %% [markdown]
# # 🧠 Noema-JEPA v1: 15M Parameter World Model with ThinkJEPA vLLM Bridge
# 
# **Architecture Highlights:**
# - **Target Parameter Budget:** Exactly **~15.0M trainable parameters** (calibrated at 14.998M).
# - **Spatial Patch Vision + Slot Attention:** Discovers discrete physical object slots from image patches.
# - **Relational C-JEPA Object Predictor:** Predicts occluded object representations using relational FlashAttention.
# - **ZeroInitResidualFiLM:** Conditions object dynamics on global VLM reasoning context via smooth residual gates.
# - **VICReg Loss (fp32 stabilized):** Protects against representation collapse with Variance hinge & Covariance decorrelation.
# - **Variance Watchdog:** Active real-time safety system detecting representation collapse and dynamically damping FiLM scale.
# - **2-Stage Colab Pipeline:** Run vLLM in Stage 1 -> cache embeddings to disk -> clear VRAM -> Train Stage 2 at maximum speed on free GPU!

# %% [markdown]
# ### 📦 Cell 1: Environment Setup in Google Colab

# %%
# Install required packages (Uncomment if running on fresh Google Colab instance)
# !pip install torch torchvision einops safetensors tqdm matplotlib
# !pip install vllm transformers accelerate  # Optional: for live vLLM inference on GPU

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# %% [markdown]
# ### 🏗️ Cell 2: Architecture Verification (~15.0M Parameters)

# %%
from noema_jepa.config import NoemaConfig
from noema_jepa.models.noema_jepa import NoemaJEPA
from noema_jepa.losses.vicreg import NoemaWorldModelLoss
from noema_jepa.monitoring.variance_watchdog import VarianceWatchdog
from noema_jepa.think_branch.vllm_extractor import VLLMThinkExtractor, DiskEmbeddingCache

# Initialize Master Configuration
config = NoemaConfig(
    img_size=128,
    embed_dim=288,
    encoder_depth=5,
    predictor_depth=6,
    num_slots=8,
    num_masked_slots=4,
    vlm_dim=1536,
    vicreg_proj_dim=768,
    batch_size=16,
)

# Instantiate Noema-JEPA World Model
model = NoemaJEPA(config)
breakdown = model.parameter_breakdown(print_table=True)

# %% [markdown]
# ### ⚡ Stage 1: ThinkJEPA Semantic Extraction with vLLM & Disk Caching
# In Google Colab, GPU VRAM is precious. To avoid OOM errors:
# 1. We extract reasoning embeddings from the VLM once.
# 2. Store them onto disk using fast `safetensors`.
# 3. Call `extractor.release_gpu_memory()` to free 100% of GPU memory for training.

# %%
# Synthetic prompts representing physical scene descriptions / reasoning
reasoning_prompts = [
    "A red sphere is moving toward a stationary blue cylinder on a frictionless surface.",
    "Two blocks colliding elastically; the heavier cube transfers momentum to the lighter block.",
    "A falling pyramid bounces against an inclined wedge and rotates clockwise.",
    "Three objects interacting under gravity with dynamic occlusions.",
]
sample_ids = [f"scene_{i:04d}" for i in range(len(reasoning_prompts))]

# Initialize Think Extractor (falls back cleanly to Mock if vLLM/GPU is absent)
cache_dir = "./cache/colab_think_embeddings"
think_extractor = VLLMThinkExtractor(config=config, cache_dir=cache_dir)

print(f"Extracting and caching ThinkJEPA embeddings for {len(sample_ids)} scenes...")
cached_embeddings = think_extractor.extract_and_cache(
    sample_ids=sample_ids,
    prompts=reasoning_prompts,
    use_cache=True,
)
print(f"Extracted shape: {cached_embeddings.shape} [B, vlm_dim]")

# Crucial Colab Step: Free all VRAM before launching Noema-JEPA training!
think_extractor.release_gpu_memory()

# %% [markdown]
# ### 🚀 Stage 2: Fast Noema-JEPA Training with AMP & Variance Watchdog

# %%
from noema_jepa.train.colab_trainer import ColabTrainer

class SyntheticPhysicalDataset(Dataset):
    """Synthetic dataset simulating physical scenes and their pre-cached ThinkJEPA vectors."""
    def __init__(self, num_samples: int = 128, img_size: int = 128, vlm_dim: int = 1536):
        self.num_samples = num_samples
        self.img_size = img_size
        self.vlm_dim = vlm_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Synthetic RGB frame
        img = torch.randn(3, self.img_size, self.img_size)
        # Synthetic ThinkJEPA reasoning vector
        think_emb = torch.randn(self.vlm_dim)
        return {"images": img, "think_embeddings": think_emb}

dataset = SyntheticPhysicalDataset(num_samples=64, img_size=config.img_size, vlm_dim=config.vlm_dim)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Setup Trainer with AMP & Watchdog
trainer = ColabTrainer(model=model, config=config)

print("Starting Noema-JEPA training epochs on Colab...")
for epoch in range(1, 3):
    epoch_stats = trainer.train_epoch(dataloader, epoch=epoch)
    print(f"Epoch {epoch} finished: Avg Loss = {epoch_stats['avg_loss']:.4f} | Throughput = {epoch_stats['throughput_samples_per_sec']:.1f} samples/s")

# %% [markdown]
# ### 🔍 Cell 3: Real-time Variance Watchdog Inspection

# %%
watchdog_summary = trainer.watchdog.get_summary()
print("\n" + "=" * 55)
print("          Variance Watchdog Diagnostic Summary          ")
print("=" * 55)
for k, v in watchdog_summary.items():
    print(f"{k:<30}: {v}")
print("=" * 55)

# %% [markdown]
# ### 🎯 Cell 4: Object Slot Latent Representation Diagnostics

# %%
model.eval()
with torch.no_grad():
    sample_batch = next(iter(dataloader))
    imgs = sample_batch["images"].to(trainer.device)
    thinks = sample_batch["think_embeddings"].to(trainer.device)
    
    outputs = model(imgs, think_embeddings=thinks)
    
    pred_slots = outputs["pred_masked_slots"]      # [B, num_masked, D]
    target_slots = outputs["target_masked_slots"]  # [B, num_masked, D]
    
    cos_sim = torch.nn.functional.cosine_similarity(pred_slots, target_slots, dim=-1)
    print(f"\nMean Cosine Similarity between Predicted and Target Object Slots: {cos_sim.mean().item():.4f}")
    print(f"Predicted Slots Std across dimensions: {pred_slots.std(dim=-1).mean().item():.4f}")
    print("\n✅ Noema-JEPA v1 demonstration completed successfully!")
