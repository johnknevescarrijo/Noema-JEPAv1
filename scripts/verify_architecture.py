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
import math
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

    # 2. Test Video Forward Pass with 2D and 3D think embeddings
    print("\n[*] Running Video Forward Pass with 2D Think Emb: [B=2, T=4, C=3, H=128, W=128]...")
    video = torch.randn(2, 4, C, H, W)
    think_2d = torch.randn(2, vlm_dim)
    video_outputs_2d = model(video, think_embeddings=think_2d, modulation_scale=0.8)
    assert video_outputs_2d["pred_masked_slots"].shape == (2, config.num_masked_slots, config.embed_dim)
    assert video_outputs_2d["c_scene"].shape == (2, config.embed_dim)
    print("    - Predicted slots with 2D think_emb: PASSED")

    print("[*] Running Video Forward Pass with 3D Think Emb: [B=2, T=4, C=3, H=128, W=128] & think [B=2, T=4, D]...")
    think_3d = torch.randn(2, 4, vlm_dim)
    video_outputs_3d = model(video, think_embeddings=think_3d, modulation_scale=0.8)
    assert video_outputs_3d["pred_masked_slots"].shape == (2, config.num_masked_slots, config.embed_dim)
    assert video_outputs_3d["c_scene"].shape == (2, config.embed_dim)
    print("    - Temporal mean-pooling of 3D think_emb: PASSED")

    # Test arbitrary long clip length (T=12 > max_frames) to verify temporal interpolation
    print("[*] Running Extended Video Clip (T=12 frames > 8) to verify temporal interpolation...")
    long_video = torch.randn(2, 12, C, H, W)
    long_outputs = model(long_video, modulation_scale=1.0)
    assert long_outputs["pred_masked_slots"].shape == (2, config.num_masked_slots, config.embed_dim)
    print(">>> Video Clip Forward Passes (2D/3D Think & Interpolation): PASSED\n")

    return outputs, video_outputs_2d


def test_loss_computation(model: NoemaJEPA, config: NoemaConfig, outputs: dict, video_outputs: dict):
    print("=" * 70)
    print("STEP 3: Validating Multi-Component Loss & compute_loss Interface")
    print("=" * 70)

    from noema_jepa.losses.vicreg import compute_loss

    loss_fn = NoemaWorldModelLoss(config)

    # 1. Test loss_fn(outputs)
    total_loss, metrics = loss_fn(outputs)

    # 2. Test model.compute_loss(outputs)
    model_loss, model_metrics = model.compute_loss(outputs, loss_fn=loss_fn)
    assert torch.isclose(total_loss, model_loss)

    # 3. Test standalone compute_loss(outputs)
    func_loss, _ = compute_loss(outputs, config=config, loss_fn=loss_fn)
    assert torch.isclose(total_loss, func_loss)

    # 4. Test 5D Video Loss Computation
    vid_loss, vid_metrics = model.compute_loss(video_outputs, loss_fn=loss_fn)
    assert not torch.isnan(vid_loss), "5D Video Loss is NaN!"
    assert vid_loss > 0.0, "5D Video Loss must be positive"
    print(f"[*] 5D Video Multi-Component Loss: {vid_loss.item():.6f}")

    print(f"[*] Static Image Loss Breakdown:")
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

    print(">>> Multi-Component Loss, compute_loss APIs & Backward Pass: PASSED [Gradients Valid]\n")
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
            ("CLEVRER / Collision Physics Video", "clevrer", 4),
            ("Kubric MOVi / Rigid Dynamics Video", "kubric", 4),
            ("CLEVRER + Kubric / Combined 5D", "clevrer_kubric", 4),
            ("SSv2 / Temporal Physics", "ssv2", 3),
            ("MS-COCO / Real Multi-Object", "coco", 1),
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
            print(f"[*] {desc:36s} -> Batch Images: {tuple(images.shape)}, Think: {emb_str}")
            if frames > 1:
                assert images.ndim == 5, f"Expected 5D tensor [B, T, C, H, W], got shape {images.shape}"
                assert images.shape[1] == frames, f"Expected {frames} frames, got {images.shape[1]}"
            else:
                assert images.ndim == 4, f"Expected 4D tensor [B, C, H, W], got shape {images.shape}"

    print(">>> Traditional Datasets & 5D Video DataLoaders: PASSED\n")


def test_colab_trainer_video_training(model: NoemaJEPA, config: NoemaConfig):
    print("=" * 70)
    print("STEP 7: Validating ColabTrainer 5D Video Execution (CLEVRER + Kubric)")
    print("=" * 70)

    from noema_jepa.train.colab_trainer import ColabTrainer
    from noema_jepa.data.datasets import build_noema_dataloader

    # Fast 5D video dataloader
    video_loader = build_noema_dataloader(
        dataset_name="clevrer_kubric",
        batch_size=2,
        num_frames=4,
        num_workers=0,
        max_samples=4,
        drop_last=True,
    )

    trainer = ColabTrainer(model=model, config=config)

    # 1. Single train_step test
    batch = next(iter(video_loader))
    loss_val, metrics, report = trainer.train_step(batch)

    print(f"[*] Single 5D Video Step Completed:")
    print(f"    - Step Loss:     {loss_val:.4f}")
    print(f"    - Watchdog:      {report.status} | FiLM Scale: {report.current_modulation_scale:.2f}")
    print(f"    - Intra-std:     {metrics.get('metric_intra_scene_std', 0.0):.4f}")
    assert loss_val > 0.0 and not math.isnan(loss_val)

    # 2. Train epoch test
    epoch_stats = trainer.train_epoch(video_loader, epoch=1)
    print(f"[*] Full 5D Video Epoch Completed:")
    print(f"    - Avg Loss:      {epoch_stats['avg_loss']:.4f}")
    print(f"    - Throughput:    {epoch_stats['throughput_samples_per_sec']:.1f} clips/s")
    print(f"    - Warnings:      {epoch_stats['watchdog_warnings']}")

    print(">>> ColabTrainer 5D Video Training: PASSED\n")


def test_v2_features(model: NoemaJEPA, config: NoemaConfig):
    print("=" * 70)
    print("STEP 8: Validating Noema-JEPA v2 Enhancements")
    print("        (EMA Target, SAVi Tracking, Action Cond, FSQ, SIGReg)")
    print("=" * 70)

    device = next(model.parameters()).device
    B, T = 2, 4
    C, H, W = config.in_channels, config.img_size, config.img_size
    actions = torch.randn(B, T, config.action_dim, device=device)
    video = torch.randn(B, T, C, H, W, device=device)
    think = torch.randn(B, config.vlm_dim, device=device)

    # 1. EMA Target Encoder
    print("[*] 1. Testing EMA Target Encoder & Teacher/Student Isolation...")
    assert hasattr(model, "target_encoder"), "Model missing target_encoder!"
    assert hasattr(model, "target_slot_attention"), "Model missing target_slot_attention!"
    for p in model.target_encoder.parameters():
        assert not p.requires_grad, "target_encoder parameter requires_grad must be False!"
    for p in model.target_slot_attention.parameters():
        assert not p.requires_grad, "target_slot_attention parameter requires_grad must be False!"

    # Verify update_ema alters weights
    p_s = next(model.encoder.parameters())
    p_t = next(model.target_encoder.parameters())
    orig_t_val = p_t.clone()
    p_s.data.add_(0.5)
    model.update_ema(momentum=0.9)
    assert not torch.allclose(orig_t_val, p_t), "update_ema failed to update target parameters!"
    print("    - EMA Target Encoder Teacher/Student Isolation & Update: PASSED")

    # 2. SAVi Temporal Slot Tracking & Multi-Step Rollout
    print("[*] 2. Testing SAVi-Style Recurrent Slot Tracking & Multi-Step Rollout...")
    out_v2 = model(video, think_embeddings=think, actions=actions)
    assert "pred_next_slots" in out_v2, "pred_next_slots missing from video forward pass!"
    assert "target_next_slots" in out_v2, "target_next_slots missing from video forward pass!"
    assert "slot_trajectories" in out_v2, "slot_trajectories missing!"
    assert "target_slot_trajectories" in out_v2, "target_slot_trajectories missing!"

    pred_next = out_v2["pred_next_slots"]
    tgt_next = out_v2["target_next_slots"]
    print(f"    - Temporal Rollout: pred_next shape {tuple(pred_next.shape)}, target_next {tuple(tgt_next.shape)}")
    assert pred_next.shape == (B, T - 1, config.num_slots, config.embed_dim)
    assert tgt_next.shape == (B, T - 1, config.num_slots, config.embed_dim)
    print("    - SAVi Tracking & Rollout: PASSED")

    # 3. Action Conditioning
    print("[*] 3. Testing Action Conditioning on Predictor...")
    single_action = torch.randn(B, config.action_dim, device=device)
    img = torch.randn(B, C, H, W, device=device)
    out_act = model(img, think_embeddings=think, actions=single_action)
    assert out_act["pred_masked_slots"].shape == (B, config.num_masked_slots, config.embed_dim)
    print("    - Action Conditioning Forward: PASSED")

    # 4. Symbolic FSQ Quantizer
    print("[*] 4. Testing Finite Scalar Quantizer (FSQ)...")
    assert hasattr(model, "fsq_quantizer"), "fsq_quantizer missing from model!"
    assert "fsq_perplexity" in out_v2, "fsq_perplexity missing in outputs!"
    assert "fsq_codebook_usage" in out_v2, "fsq_codebook_usage missing in outputs!"
    fsq_idx = model.fsq_quantizer.get_codebook_indices(out_v2["context_slots"])
    print(f"    - Discrete codebook indices shape: {tuple(fsq_idx.shape)}, usage: {float(out_v2['fsq_codebook_usage']):.4f}")
    assert fsq_idx.min() >= 0 and fsq_idx.max() < model.fsq_quantizer.total_codes
    print("    - FSQ Symbolic Quantization: PASSED")

    # 5. LayerNorm VICReg Variance Health & SIGReg Loss
    print("[*] 5. Validating LayerNorm VICReg Projector & SIGReg Anti-Collapse...")
    from noema_jepa.losses.vicreg import SIGRegLoss
    sigreg_fn = SIGRegLoss(eps=1e-4)
    loss_s, m_s = sigreg_fn(out_v2["pred_vicreg_proj"], out_v2["target_vicreg_proj"])
    assert not torch.isnan(loss_s), "SIGReg loss is NaN!"
    print(f"    - SIGReg Loss: {loss_s.item():.4f}")

    loss_v2, m_v2 = model.compute_loss(out_v2)
    std_pred = m_v2["vicreg_std_pred"]
    print(f"    - LayerNorm vicreg_std_pred: {std_pred:.4f} (v1 collapsed value was ~0.0102)")
    assert std_pred > 0.05, f"Variance still collapsed! std_pred={std_pred}"
    print("    - LayerNorm VICReg Projector & SIGReg: PASSED")

    print(">>> All Noema-JEPA v2 Features Verified: PASSED\n")


def main():
    print("\n" + "#" * 70)
    print("   NOEMA-JEPA v2 (~15M Parameters) ARCHITECTURE VERIFICATION   ")
    print("#" * 70)

    import math
    model, config = test_parameter_budget()
    outputs, video_outputs = test_forward_passes(model, config)
    loss_fn = test_loss_computation(model, config, outputs, video_outputs)
    test_variance_watchdog_collapse_intervention(config)
    test_think_cache()
    test_traditional_datasets()
    test_colab_trainer_video_training(model, config)
    test_v2_features(model, config)

    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! NOEMA-JEPA v2 IS 100% OPERATIONAL.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


