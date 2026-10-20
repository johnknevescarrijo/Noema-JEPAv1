#!/usr/bin/env python3
"""
Verification Script for Noema-JEPA v1 Architecture.

Validates:
1. Trainable parameter count target (~15.0M params).
2. Forward pass with image [B, C, H, W] and video [B, T, C, H, W].
3. Full multi-component loss evaluation (C-JEPA, VICReg fp32, Intra-scene, ThinkJEPA).
4. VarianceWatchdog real-time anti-collapse reaction simulation.
"""

import sys
import os
from pathlib import Path
import torch

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noema_jepa.config import NoemaConfig
from noema_jepa.models.noema_jepa import NoemaJEPA
from noema_jepa.losses.vicreg import NoemaWorldModelLoss
from noema_jepa.monitoring.variance_watchdog import VarianceWatchdog
from noema_jepa.think_branch.vllm_extractor import DiskEmbeddingCache, MockThinkExtractor


def test_parameter_budget():
    print("\n" + "=" * 70)
    print("STEP 1: Validating Trainable Parameter Budget (~15.0M Target)")
    print("=" * 70)

    config = NoemaConfig()
    model = NoemaJEPA(config)

    breakdown = model.parameter_breakdown(print_table=True)
    total_trainable = breakdown["Total"]

    # Parameter target: ~15M (acceptable range 14.5M to 15.5M)
    lower_bound = 14_500_000
    upper_bound = 15_500_000

    print(f"[*] Trainable parameters: {total_trainable:,}")
    print(f"[*] Budget tolerance window: [{lower_bound:,} - {upper_bound:,}]")

    assert lower_bound <= total_trainable <= upper_bound, (
        f"FAILED: Model has {total_trainable:,} parameters, which falls outside [{lower_bound:,}, {upper_bound:,}]!"
    )
    print(">>> Parameter Budget Check: PASSED [15.0M Target Satisfied]\n")
    return model, config


def test_forward_passes(model: NoemaJEPA, config: NoemaConfig):
    print("=" * 70)
    print("STEP 2: Validating Forward Pass (Images & Video Clips)")
    print("=" * 70)

    B = 4
    C, H, W = config.in_channels, config.img_size, config.img_size
    vlm_dim = config.vlm_dim

    # 1. Test Static Image Forward Pass
    print("[*] Running Image Forward Pass: [B=4, C=3, H=128, W=128]...")
    images = torch.randn(B, C, H, W)
    think_embs = torch.randn(B, vlm_dim)

    outputs = model(images, think_embeddings=think_embs, modulation_scale=1.0)

    assert "pred_masked_slots" in outputs, "Missing pred_masked_slots in outputs"
    assert "target_masked_slots" in outputs, "Missing target_masked_slots in outputs"
    assert "pred_vicreg_proj" in outputs, "Missing pred_vicreg_proj in outputs"
    assert "pred_think_proj" in outputs, "Missing pred_think_proj in outputs"

    pred_slots = outputs["pred_masked_slots"]
    target_slots = outputs["target_masked_slots"]

    print(f"    - Predicted masked slots shape: {tuple(pred_slots.shape)} (expected [{B}, {config.num_masked_slots}, {config.embed_dim}])")
    print(f"    - Target masked slots shape:    {tuple(target_slots.shape)}")
    print(f"    - VICReg projection shape:      {tuple(outputs['pred_vicreg_proj'].shape)}")
    print(f"    - Think alignment shape:        {tuple(outputs['pred_think_proj'].shape)}")

    assert pred_slots.shape == (B, config.num_masked_slots, config.embed_dim)
    assert not torch.isnan(pred_slots).any(), "NaN found in predicted slots!"
    print(">>> Static Image Forward Pass: PASSED")

    # 2. Test Short Video Forward Pass
    print("\n[*] Running Video Forward Pass: [B=2, T=3, C=3, H=128, W=128]...")
    video = torch.randn(2, 3, C, H, W)
    video_outputs = model(video, modulation_scale=0.8)
    assert video_outputs["pred_masked_slots"].shape == (2, config.num_masked_slots, config.embed_dim)
    print(">>> Video Clip Forward Pass: PASSED\n")

    return outputs


def test_loss_computation(model: NoemaJEPA, config: NoemaConfig, outputs: dict):
    print("=" * 70)
    print("STEP 3: Validating Multi-Component Loss (VICReg fp32 + C-JEPA + Intra-Scene)")
    print("=" * 70)

    loss_fn = NoemaWorldModelLoss(config)
    total_loss, metrics = loss_fn(outputs)

    print(f"[*] Loss Breakdown:")
    for k, v in metrics.items():
        print(f"    - {k:<26}: {v:.6f}")

    assert not torch.isnan(total_loss), "Loss computed to NaN!"
    assert not torch.isinf(total_loss), "Loss computed to Inf!"
    assert total_loss > 0.0, "Loss must be positive"

    # Backward pass check to ensure gradient connectivity
    total_loss.backward()

    # Verify gradients in key modules
    assert model.encoder.patch_embed.weight.grad is not None, "Encoder gradient is None!"
    assert model.slot_attention.project_q.weight.grad is not None, "Slot Attention gradient is None!"
    assert model.predictor.pred_head[0].weight.grad is not None, "Predictor gradient is None!"
    assert model.think_bridge.projector[0].weight.grad is not None, "Think Bridge gradient is None!"

    print(">>> Multi-Component Loss & Backward Pass: PASSED [Gradients Valid]\n")
    return loss_fn


def test_variance_watchdog_collapse_intervention(config: NoemaConfig):
    print("=" * 70)
    print("STEP 4: Simulating VarianceWatchdog Anti-Collapse Intervention")
    print("=" * 70)

    watchdog = VarianceWatchdog(config)
    loss_fn = NoemaWorldModelLoss(config)

    # 1. Healthy step
    healthy_metrics = {
        "vicreg_std_pred": 1.05,
        "metric_intra_scene_std": 0.28,
    }
    report1 = watchdog.step(healthy_metrics, loss_fn=loss_fn)
    print(f"[Step 1 - Healthy State]")
    print(f"    Status: {report1.status} | FiLM Scale: {report1.current_modulation_scale:.2f} | Think W: {report1.current_think_weight:.4f}")
    assert report1.status in ("HEALTHY", "STABLE")

    # 2. Simulated collapse: intra_scene_std collapses to 0.01 (FiLM overshadowing slots)
    print(f"\n[Step 2 - Simulated FiLM Collapse: intra_scene_std=0.01 (Threshold={config.watchdog_std_threshold})]")
    collapsed_metrics = {
        "vicreg_std_pred": 0.95,
        "metric_intra_scene_std": 0.01,  # Severe collapse
    }
    report2 = watchdog.step(collapsed_metrics, loss_fn=loss_fn)
    print(f"    Status: {report2.status} | FiLM Scale: {report2.current_modulation_scale:.3f} | Think W: {report2.current_think_weight:.4f}")
    print(f"    Action: {report2.action_taken}")

    assert report2.status in ("CRITICAL", "WARNING"), f"Expected collapse detection, got {report2.status}"
    assert report2.current_modulation_scale < config.initial_film_modulation_scale, "Watchdog failed to damp FiLM scale!"
    assert report2.current_think_weight < config.think_loss_weight, "Watchdog failed to damp ThinkJEPA loss weight!"

    # 3. Sustained damping
    report3 = watchdog.step(collapsed_metrics, loss_fn=loss_fn)
    print(f"\n[Step 3 - Sustained Collapse]")
    print(f"    Status: {report3.status} | FiLM Scale: {report3.current_modulation_scale:.3f} | Think W: {report3.current_think_weight:.4f}")
    assert report3.current_modulation_scale < report2.current_modulation_scale

    # 4. Recovery: Slots expand again
    print(f"\n[Step 4 - Representation Restored: intra_scene_std=0.22]")
    recovered_metrics = {
        "vicreg_std_pred": 1.10,
        "metric_intra_scene_std": 0.22,
    }
    report4 = watchdog.step(recovered_metrics, loss_fn=loss_fn)
    print(f"    Status: {report4.status} | FiLM Scale: {report4.current_modulation_scale:.3f} | Think W: {report4.current_think_weight:.4f}")
    assert report4.status == "HEALTHY"
    assert report4.current_modulation_scale > report3.current_modulation_scale, "Watchdog failed to restore scale!"

    print(">>> VarianceWatchdog Anti-Collapse Mechanism: PASSED\n")


def test_think_cache():
    print("=" * 70)
    print("STEP 5: Validating ThinkJEPA Disk Embedding Cache")
    print("=" * 70)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskEmbeddingCache(cache_dir=tmpdir)
        extractor = MockThinkExtractor(vlm_dim=1536)

        prompts = [
            "Predict the physical trajectory of the falling red cube.",
            "Analyze collision dynamics between the sphere and cylinder.",
        ]
        sample_ids = ["sample_001", "sample_002"]

        embs = extractor.extract_text(prompts)
        assert embs.shape == (2, 1536)

        cache.batch_save(sample_ids, embs)
        assert cache.has("sample_001")
        assert cache.has("sample_002")
        assert cache.count() == 2

        loaded = cache.load("sample_001")
        assert torch.allclose(loaded, embs[0], atol=1e-5)
        print(f"[*] Successfully saved and verified 2 embeddings in cache ({tmpdir}).")

    print(">>> ThinkJEPA Embedding Cache: PASSED\n")


def test_traditional_datasets():
    print("=" * 70)
    print("STEP 6: Validating Traditional Benchmarks & DataLoaders (CLEVR, SSv2, COCO, Kubric)")
    print("=" * 70)

    import tempfile
    from noema_jepa.data.datasets import build_noema_dataloader

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-seed cache to test embedding attachment
        cache = DiskEmbeddingCache(cache_dir=tmpdir)
        for i in range(4):
            cache.save(f"synthetic_train_{i:06d}", torch.randn(1536))

        benchmarks = [
            ("CLEVR / Slot Attention", "clevr", 1),
            ("SSv2 / Temporal Physics", "ssv2", 3),
            ("MS-COCO / Real Multi-Object", "coco", 1),
            ("Kubric MOVi / Dynamic Collisions", "kubric", 2),
            ("Synthetic Toy / Procedural", "synthetic", 1),
        ]

        for desc, name, frames in benchmarks:
            loader = build_noema_dataloader(
                dataset_name=name,
                batch_size=2,
                num_frames=frames,
                cache_dir=tmpdir if name == "synthetic" else None,
                num_workers=0,
                max_samples=4,
            )
            batch = next(iter(loader))
            images = batch["images"]
            think_embs = batch.get("think_embeddings")
            emb_str = f"cached {tuple(think_embs.shape)}" if think_embs is not None else "None (unconditioned)"
            print(f"[*] {desc:34s} -> Batch Images: {tuple(images.shape)}, Think: {emb_str}")
            if frames > 1:
                assert images.ndim == 5, f"Expected 5D tensor [B, T, C, H, W], got shape {images.shape}"
            else:
                assert images.ndim == 4, f"Expected 4D tensor [B, C, H, W], got shape {images.shape}"

    print(">>> Traditional Datasets & DataLoaders: PASSED\n")


def main():
    print("\n" + "#" * 70)
    print("   NOEMA-JEPA v1 (~15M Parameters) ARCHITECTURE VERIFICATION   ")
    print("#" * 70)

    model, config = test_parameter_budget()
    outputs = test_forward_passes(model, config)
    loss_fn = test_loss_computation(model, config, outputs)
    test_variance_watchdog_collapse_intervention(config)
    test_think_cache()
    test_traditional_datasets()

    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! ARCHITECTURE IS 100% READY.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

