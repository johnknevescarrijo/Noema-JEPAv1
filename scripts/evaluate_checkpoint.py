#!/usr/bin/env python3
"""
Evaluation and Inference Script for Trained Noema-JEPA v1 Checkpoints.

Usage:
    python3 scripts/evaluate_checkpoint.py
    python3 scripts/evaluate_checkpoint.py --checkpoint checkpoints/noema_jepa_video_best.pt --dataset clevrer_kubric --batches 10
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noema_jepa.config import NoemaConfig
from noema_jepa.models.noema_jepa import NoemaJEPA
from noema_jepa.losses.vicreg import NoemaWorldModelLoss
from noema_jepa.data.datasets import build_noema_dataloader


def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/noema_jepa_video_best.pt",
    dataset_name: str = "clevrer_kubric",
    num_batches: int = 5,
    batch_size: int = 4,
    device: str = None,
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"❌ Error: Checkpoint '{checkpoint_path}' not found!")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(f"  NOEMA-JEPA v1: CHECKPOINT EVALUATION & INFERENCE TEST")
    print("=" * 70)
    print(f"[*] Checkpoint: {ckpt_file.resolve()}")
    print(f"[*] Device:     {device.upper()}")
    print(f"[*] Benchmark:  {dataset_name}")
    print(f"[*] Batches:    {num_batches} (Batch size: {batch_size})")

    # Load checkpoint
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    step = ckpt.get("step", "N/A")
    watchdog_saved = ckpt.get("watchdog_summary", {})
    print(f"[*] Resumed Step: {step}")
    if watchdog_saved:
        print(f"[*] Checkpoint Watchdog Info: Warnings={watchdog_saved.get('collapse_warnings_total', 0)}, FiLM Scale={watchdog_saved.get('current_modulation_scale', 1.0)}")

    config = NoemaConfig()
    model = NoemaJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_fn = NoemaWorldModelLoss(config).to(device)

    # Build evaluation dataloader
    val_loader = build_noema_dataloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_frames=config.num_frames,
        split="test",
        num_workers=0,
        max_samples=num_batches * batch_size,
        drop_last=False,
    )

    total_loss = 0.0
    total_cos_sim = 0.0
    total_slot_std = 0.0
    total_intra_scene_std = 0.0
    metrics_accum = {}
    batches_processed = 0

    print("\n[*] Running evaluation over batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            images = batch["images"].to(device)
            think_embs = batch.get("think_embeddings")
            if think_embs is not None:
                think_embs = think_embs.to(device)

            outputs = model(images, think_embeddings=think_embs, modulation_scale=1.0)
            loss, metrics = loss_fn(outputs)

            pred_slots = outputs["pred_masked_slots"]      # [B, num_masked, D]
            target_slots = outputs["target_masked_slots"]  # [B, num_masked, D]

            # Cosine similarity between predicted and target latent slots
            cos_sim = F.cosine_similarity(pred_slots, target_slots, dim=-1).mean().item()
            slot_std = pred_slots.std(dim=-1).mean().item()
            intra_std = metrics.get("metric_intra_scene_std", 0.0)

            total_loss += loss.item()
            total_cos_sim += cos_sim
            total_slot_std += slot_std
            total_intra_scene_std += intra_std

            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + (v if isinstance(v, float) else v.item())

            batches_processed += 1
            print(f"  [Batch {batches_processed:02d}/{num_batches:02d}] Loss: {loss.item():.4f} | CosSim: {cos_sim:.4f} | SlotStd: {slot_std:.4f} | IntraStd: {intra_std:.4f}")

    if batches_processed == 0:
        print("❌ Error: No batches could be loaded for evaluation.")
        return

    avg_loss = total_loss / batches_processed
    avg_cos_sim = total_cos_sim / batches_processed
    avg_slot_std = total_slot_std / batches_processed
    avg_intra_std = total_intra_scene_std / batches_processed

    print("\n" + "=" * 70)
    print(f"                 NOEMA-JEPA {getattr(config, 'model_version', 'v2').upper()} EVALUATION REPORT")
    print("=" * 70)
    print(f"  Avg Multi-Component Loss:                    {avg_loss:.4f}")
    print(f"  Avg Slot Cosine Similarity (Pred vs Target): {avg_cos_sim:.4f}  ({'PASSED (High Fidelity)' if avg_cos_sim > 0.80 else 'ACCEPTABLE'})")
    print(f"  Avg Slot Latent Diversity (Std Dev):         {avg_slot_std:.4f}  ({'HIGH DIVERSITY' if avg_slot_std > 2.0 else 'NORMAL'})")
    print(f"  Avg Intra-Scene Slot Separation Std:         {avg_intra_std:.4f}  ({'NO COLLAPSE' if avg_intra_std > config.watchdog_std_threshold else 'COLLAPSE RISK'})")
    if "vicreg_std_pred" in metrics_accum:
        avg_vic_std = metrics_accum["vicreg_std_pred"] / batches_processed
        print(f"  Avg VICReg Latent Std (Pred):                {avg_vic_std:.4f}  ({'HEALTHY' if avg_vic_std > 0.20 else 'SHRINKAGE'})")
    if "fsq_codebook_usage" in metrics_accum:
        avg_fsq_use = metrics_accum["fsq_codebook_usage"] / batches_processed
        avg_fsq_ppl = metrics_accum.get("fsq_perplexity", 0.0) / batches_processed
        print(f"  FSQ Symbolic Codebook Usage:                 {avg_fsq_use * 100:.2f}% (Perplexity: {avg_fsq_ppl:.2f})")
    print("-" * 70)
    print("  Detailed Loss Component Averages:")
    for k, v in metrics_accum.items():
        print(f"    - {k:<28}: {v / batches_processed:.6f}")
    print("=" * 70)
    print("✅ Model inference and evaluation completed successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Noema-JEPA v1 checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/noema_jepa_video_best.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="clevrer_kubric", help="Dataset/benchmark to test on")
    parser.add_argument("--batches", type=int, default=5, help="Number of batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        num_batches=args.batches,
        batch_size=args.batch_size,
        device=args.device,
    )
