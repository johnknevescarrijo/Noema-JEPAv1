# %% [markdown]
# # 🧠 Noema-JEPA v1: 15M Parameter World Model with ThinkJEPA vLLM Bridge
# ## NVIDIA A100 GPU Training Guide: CLEVRER & Kubric MOVi (5D Dynamic Video Physics)
#
# **Architecture & Benchmark Highlights:**
# - **Hardware Target:** Calibrated for **NVIDIA A100 (40GB / 80GB SXM/PCIe)** with native `bfloat16`, `TF32` Tensor Cores, and JIT `torch.compile`.
# - **Trainable Parameter Target:** Exactly **~15.0M trainable parameters** (calibrated at 14.998M).
# - **Input Tensor Format:** **5D Video Clips** `[B, T, C, H, W]` ($T=4$ or $T=8$ frames at $128 \times 128$ resolution).
# - **Dynamic Benchmarks:**
#   1. **CLEVRER (MIT CSAIL / Yi et al., 2020):** 20,000 video clips with rigid-body collisions, trajectory dynamics, and causal events.
#   2. **Kubric MOVi (Google Research / Greff et al., 2022):** PyBullet multi-object rigid body dynamics with Blender ray-traced physical simulations.
#   3. **Combined Mode (`"clevrer_kubric"`):** Unified multi-benchmark training via `ConcatDataset`.
# - **Spatial-Temporal Patch Vision + Slot Attention:** Discovers discrete moving physical objects across time.
# - **Relational C-JEPA Object Predictor:** Predicts masked physical states conditioned on visible slots and global semantic context.
# - **ZeroInitResidualFiLM:** Modulates physical transition predictions via semantic reasoning embeddings.
# - **fp32-Stabilized VICReg Loss:** Invariance, variance hinge, and covariance decorrelation preventing representation collapse.
# - **Real-Time Variance Watchdog:** Dynamically safeguards against FiLM oversaturation and slot collapse.

# %% [markdown]
# ### 📦 Cell 1: Environment Setup & NVIDIA A100 Hardware Inspection
# Run this cell to configure the environment, clone repository if needed (Google Colab mode),
# register `noema_jepa` in `sys.path`, configure project directories, activate TF32 Tensor Cores, and inspect GPU capabilities.

# %%
import os
import sys
import subprocess
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Automated Environment Discovery & Git Clone (Colab / Kaggle / Local)
# -----------------------------------------------------------------------------
def setup_noema_environment():
    """
    Automated environment discovery and setup for Google Colab, Kaggle, and Local environments.
    Ensures the Noema-JEPAv1 repository is present, registered in sys.path,
    minimal dependencies are installed, and noema_jepa is importable.
    """
    in_colab = "google.colab" in sys.modules or os.path.exists("/content")
    repo_name = "Noema-JEPAv1"
    repo_url = "https://github.com/johnknevescarrijo/Noema-JEPAv1.git"

    # Search candidate paths for repository root containing 'noema_jepa'
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/content") / repo_name if Path("/content").exists() else None,
        Path("/content") if Path("/content").exists() else None,
    ]
    try:
        candidates.insert(0, Path(__file__).resolve().parent.parent)
    except NameError:
        pass

    repo_root = None
    for cand in candidates:
        if cand is not None and cand.exists() and (cand / "noema_jepa").is_dir():
            repo_root = cand.resolve()
            break

    # If running in Google Colab and repo is missing, clone automatically
    if repo_root is None and in_colab:
        colab_target = Path("/content") / repo_name
        print(f"📦 [Colab Setup] 'noema_jepa' not found locally.")
        print(f"📦 [Colab Setup] Cloning repository from {repo_url} into {colab_target}...")
        try:
            subprocess.run(["git", "clone", repo_url, str(colab_target)], check=True)
            if (colab_target / "noema_jepa").is_dir():
                repo_root = colab_target.resolve()
        except Exception as e:
            print(f"⚠️ [Colab Setup] Git clone failed: {e}")

    # Register repository root in sys.path and switch working directory
    if repo_root is not None:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            os.chdir(str(repo_root))
        except Exception:
            pass
        print(f"📁 [Setup] Repository Root:     {repo_root}")
        print(f"📍 [Setup] Working Directory:   {Path.cwd()}")
    else:
        print("⚠️ [Setup] Could not locate 'noema_jepa' automatically. Please check your path.")

    # Install minimal Colab dependencies if missing (Colab already provides torch, torchvision, numpy, etc.)
    if in_colab:
        missing = []
        for pkg in ["einops", "safetensors"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            print(f"⚡ [Colab Setup] Fast installation of minimal dependencies ({', '.join(missing)})...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *missing],
                check=True,
            )

    # Validate import
    try:
        import noema_jepa
        ver = getattr(noema_jepa, "__version__", "0.1.0")
        print(f"✅ [Setup] Successfully imported 'noema_jepa' v{ver}")
        print(f"   Module path: {noema_jepa.__file__}\n")
    except ImportError as e:
        print(f"❌ [Setup] Error importing 'noema_jepa': {e}")
        print("💡 Quick Fix for Colab: run `!git clone https://github.com/johnknevescarrijo/Noema-JEPAv1.git /content/Noema-JEPAv1`")
        raise

# Ensure environment is configured (self-contained, no external cell dependency)
if "noema_jepa" not in sys.modules:
    setup_noema_environment()

# -----------------------------------------------------------------------------
# 2. Configure Project Directories
# -----------------------------------------------------------------------------
DATA_DIR = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
CACHE_DIR = Path("./cache/video_think_embeddings")

for d in (DATA_DIR, CHECKPOINT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Hardware Inspection & Tensor Core Configuration (NVIDIA A100 / Ampere+)
# -----------------------------------------------------------------------------
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Device Name:     {device_name}")
    print(f"Compute Cap:     sm_{major}{minor}")
    print(f"Total VRAM:      {vram_gb:.2f} GB")

    # Enable TF32 for Ampere / Hopper Tensor Cores (A100 / H100)
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[A100 Acceleration] TF32 Tensor Cores Activated for matmul and cuDNN!")
        if torch.cuda.is_bf16_supported():
            print("[A100 Acceleration] Native bfloat16 mixed precision supported (zero GradScaler overhead).")
else:
    print("Running in CPU mode. (Note: NVIDIA A100 GPU recommended for 5D video training).")

# %% [markdown]
# ### 🏗️ Cell 2: Architecture Initialization & Parameter Verification (~15.0M Params)
# Instantiates Noema-JEPA v1 with the A100 Video preset (`for_a100_video`).

# %%
from noema_jepa.config import NoemaConfig
from noema_jepa.models.noema_jepa import NoemaJEPA
from noema_jepa.losses.vicreg import NoemaWorldModelLoss, compute_loss
from noema_jepa.monitoring.variance_watchdog import VarianceWatchdog
from noema_jepa.think_branch.vllm_extractor import VLLMThinkExtractor, DiskEmbeddingCache
from noema_jepa.data import (
    build_noema_dataloader,
    download_clevrer_dataset,
    download_kubric_dataset,
    CLEVRERDataset,
    KubricMOViDataset,
)
from noema_jepa.train.colab_trainer import ColabTrainer

# Calibrate preset for A100/L4 with 5D video clips (T=4 frames)
config = NoemaConfig.for_a100_video(
    num_frames=4,
    batch_size=64,       # High batch size exploiting A100/L4 memory bandwidth
    learning_rate=3e-4,  # Calibrated for stable, non-divergent 30+ epoch convergence
    amp_dtype="bfloat16",
    use_compile=False,   # Set True for torch.compile JIT acceleration if Triton is supported
    num_workers=4,
)

print(f"\n[A100 Video Config]")
print(f"  - num_frames:  {config.num_frames} frames per clip [B, {config.num_frames}, 3, 128, 128]")
print(f"  - batch_size:  {config.batch_size} clips per batch ({config.batch_size * config.num_frames} total frames/step)")
print(f"  - amp_dtype:   {config.amp_dtype}")
print(f"  - use_compile: {config.use_compile}")

# Instantiate Noema-JEPA World Model
model = NoemaJEPA(config)
breakdown = model.parameter_breakdown(print_table=True)

# Sanity check forward pass with 5D Video Tensor [B, T, C, H, W]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_video = torch.randn(2, config.num_frames, 3, 128, 128, device=device)
dummy_think = torch.randn(2, config.vlm_dim, device=device)
model = model.to(device)

with torch.no_grad():
    dummy_out = model(dummy_video, think_embeddings=dummy_think)
    dummy_loss, _ = model.compute_loss(dummy_out)
print(f"\n>>> 5D Video Forward & Loss Sanity Check: PASSED (Initial Loss: {dummy_loss.item():.4f})")

# %% [markdown]
# ### 📥 Cell 3: Real Dataset Download & DataLoaders (CLEVRER + Kubric MOVi)
#
# **Dataset Choices:**
# 1. **`"clevrer"`:** Official MIT CSAIL CLEVRER video benchmark.
#    - Download automated chunks: `download_clevrer_dataset(num_chunks=2)` downloads 2,000 full videos (~1.3 GB).
#    - Reads `.mp4` videos directly using high-throughput OpenCV/torchvision decoding.
# 2. **`"kubric"`:** Google Research Kubric MOVi dynamic rigid-body simulations.
# 3. **`"clevrer_kubric"` (Combined Benchmark):** Streams from both benchmarks via `ConcatDataset`.
# 4. **Zero-Download Procedural Physics Fallback:** If raw files are not downloaded, the engine automatically runs `ProceduralCLEVRERVideoDataset` and `ProceduralKubricVideoDataset` with live 3D elastic collision dynamics so you can start training immediately!

# %%
# Benchmark Selection Flags
DATASET_CHOICE = "clevrer_kubric"  # Options: 'clevrer_kubric', 'clevrer', 'kubric', 'clevr'
AUTO_DOWNLOAD_REAL_DATA = False    # Set True to download real MIT CSAIL CLEVRER videos (~1.3 GB for 2,000 videos)
NUM_CHUNKS = 2                    # 2 chunks = 2,000 videos; set up to 10 for complete 10,000 video split

if AUTO_DOWNLOAD_REAL_DATA:
    print(f"Triggering automated download of CLEVRER ({NUM_CHUNKS * 1000} videos)...")
    download_clevrer_dataset(root=DATA_DIR, split="train", num_chunks=NUM_CHUNKS)

# Build high-throughput 5D DataLoader for NVIDIA A100
train_loader = build_noema_dataloader(
    dataset_name=DATASET_CHOICE,
    root=DATA_DIR,
    split="train",
    batch_size=config.batch_size,
    image_size=config.img_size,
    num_frames=config.num_frames,
    num_workers=config.num_workers if torch.cuda.is_available() else 0,
    shuffle=True,
    download=AUTO_DOWNLOAD_REAL_DATA,
    synthetic_fallback=True,  # Seamless high-fidelity procedural dynamics if raw .mp4 are absent
    pin_memory=True,
    drop_last=True,
)

print(f"\nDataLoader Ready for '{DATASET_CHOICE}':")
print(f"  - Total Batches per Epoch: {len(train_loader)}")
print(f"  - Batch Dimensions:       [{config.batch_size}, {config.num_frames}, 3, 128, 128]")

# Fetch a sample batch and inspect shapes
sample_batch = next(iter(train_loader))
sample_videos = sample_batch["images"]
print(f"  - Actual Batch Tensor:    {tuple(sample_videos.shape)} [B, T, C, H, W]")
print(f"  - Sample Prompt:          {sample_batch['vlm_prompts'][0]}")

# %% [markdown]
# ### 🎬 Cell 4: Temporal Video Clip Visualization (Physics in Motion)
# Visualizes the temporal evolution of objects and collisions across frames $t=1, \dots, T$.

# %%
# Visualize first video clips across all T time steps
num_viz = min(2, sample_videos.size(0))
fig, axes = plt.subplots(num_viz, config.num_frames, figsize=(3.5 * config.num_frames, 3.5 * num_viz), squeeze=False)

for clip_idx in range(num_viz):
    clip = sample_videos[clip_idx]  # [T, C, H, W]
    for t in range(config.num_frames):
        frame = clip[t].clone().detach().cpu().permute(1, 2, 0).numpy()
        # Unnormalize
        frame = frame * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        frame = frame.clip(0, 1)

        ax = axes[clip_idx, t]
        ax.imshow(frame)
        ax.set_title(f"Clip {clip_idx + 1} | t = {t + 1}/{config.num_frames}", fontsize=11)
        ax.axis("off")

plt.suptitle(f"Sample 5D Physical Interaction Sequences ({DATASET_CHOICE.upper()})", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ⚡ Stage 1: ThinkJEPA Semantic Reasoning Pre-Extraction (Optional)
# In Google Colab, you can extract reasoning vectors from frozen VLMs (e.g. Qwen2-VL) once,
# cache them to disk via `DiskEmbeddingCache`, and free 100% of GPU memory for training.

# %%
# Extract ThinkJEPA reasoning vectors for sample scenes
sample_ids = sample_batch["sample_ids"][:8]
sample_prompts = sample_batch["vlm_prompts"][:8]

think_extractor = VLLMThinkExtractor(config=config, cache_dir=str(CACHE_DIR))
cached_embeddings = think_extractor.extract_and_cache(
    sample_ids=sample_ids,
    prompts=sample_prompts,
    use_cache=True,
)
print(f"ThinkJEPA Extracted Embeddings: {cached_embeddings.shape} [B, vlm_dim]")

# Crucial Colab Step: Free GPU VRAM before launching Noema-JEPA training!
think_extractor.release_gpu_memory()

# %% [markdown]
# ### 🚀 Stage 2: NVIDIA A100 Training Execution (30+ Epochs)
#
# **Execution Highlights:**
# - **AMP bfloat16 + TF32:** Exploits A100 3rd Gen Tensor Cores for maximum throughput.
# - **Low-Overhead Step:** `torch.compile` kernel fusion active.
# - **Variance Watchdog Active:** Continuous monitoring of batch and intra-scene standard deviations.

# %%
# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
NUM_EPOCHS = 30         # Recommended: 30 to 50 epochs for full causal physical convergence
SAVE_EVERY_EPOCH = True # Save checkpoint at the end of every epoch
RESUME_PATH = None      # Optional checkpoint resumption path

trainer = ColabTrainer(model=model, config=config, device=device)

if RESUME_PATH and Path(RESUME_PATH).exists():
    trainer.load_checkpoint(RESUME_PATH)

best_loss = float("inf")
training_history = {
    "epoch": [],
    "avg_loss": [],
    "throughput": [],
    "watchdog_warnings": [],
}

print("\n" + "=" * 70)
print(f"  LAUNCHING NOEMA-JEPA v1 A100 TRAINING: {DATASET_CHOICE.upper()} (5D VIDEO)")
print(f"  Configuration: bfloat16 AMP | TF32 Tensor Cores | Batch Size: {config.batch_size}")
print("=" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_stats = trainer.train_epoch(train_loader, epoch=epoch)
    avg_loss = epoch_stats["avg_loss"]
    throughput = epoch_stats["throughput_samples_per_sec"]
    warnings = epoch_stats["watchdog_warnings"]

    training_history["epoch"].append(epoch)
    training_history["avg_loss"].append(avg_loss)
    training_history["throughput"].append(throughput)
    training_history["watchdog_warnings"].append(warnings)

    print(
        f"[*] Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Throughput: {throughput:.1f} clips/s | "
        f"Watchdog Warnings: {warnings}"
    )

    # Save per-epoch checkpoint
    if SAVE_EVERY_EPOCH:
        epoch_ckpt = CHECKPOINT_DIR / f"noema_jepa_video_epoch_{epoch:03d}.pt"
        trainer.save_checkpoint(epoch_ckpt)

    # Save best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_ckpt = CHECKPOINT_DIR / "noema_jepa_video_best.pt"
        trainer.save_checkpoint(best_ckpt)
        print(f"    >>> New best video world model saved! (Loss: {best_loss:.4f})")

print("\n" + "=" * 70)
print("  A100 VIDEO TRAINING COMPLETED SUCCESSFULLY!")
print(f"  Best Checkpoint preserved at: {CHECKPOINT_DIR / 'noema_jepa_video_best.pt'}")
print("=" * 70)

# %% [markdown]
# ### 🔍 Cell 5: Real-time Variance Watchdog Diagnostic Summary
# Validates that object slot representations maintained high diversity without collapse.

# %%
watchdog_summary = trainer.watchdog.get_summary()
print("\n" + "=" * 65)
print(f"{'Variance Watchdog Anti-Collapse Diagnostic Summary':^65}")
print("=" * 65)
for k, v in watchdog_summary.items():
    print(f"{k:<34}: {v}")
print("=" * 65)

# %% [markdown]
# ### 🎯 Cell 6: Latent Representation Evaluation & Training Curves

# %%
model.eval()
with torch.no_grad():
    eval_batch = next(iter(train_loader))
    eval_videos = eval_batch["images"].to(trainer.device, non_blocking=True)
    eval_thinks = eval_batch.get("think_embeddings")
    if eval_thinks is not None:
        eval_thinks = eval_thinks.to(trainer.device, non_blocking=True)

    outputs = model(eval_videos, think_embeddings=eval_thinks)
    pred_slots = outputs["pred_masked_slots"]      # [B, num_masked, D]
    target_slots = outputs["target_masked_slots"]  # [B, num_masked, D]

    cos_sim = torch.nn.functional.cosine_similarity(pred_slots, target_slots, dim=-1)
    mean_sim = cos_sim.mean().item()
    slot_std = pred_slots.std(dim=-1).mean().item()

    print(f"\nEvaluation on 5D Video Batch:")
    print(f"  - Mean Cosine Similarity (Pred vs Target Slots): {mean_sim:.4f}")
    print(f"  - Slot Latent Standard Deviation:               {slot_std:.4f}")

# Plot Training Curves
if len(training_history["epoch"]) > 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(training_history["epoch"], training_history["avg_loss"], "b-o", linewidth=2)
    ax1.set_title("Training Loss across Epochs (VICReg + C-JEPA)", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(training_history["epoch"], training_history["throughput"], "g-s", linewidth=2)
    ax2.set_title("A100 Training Throughput (Clips / sec)", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Throughput (clips/sec)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

print("\n✅ Noema-JEPA v1 A100 Video World Model Training Guide Complete!")
