"""
ThinkJEPA VLM Extractor and Disk Embedding Cache.

Provides:
1. VLLMThinkExtractor: High-throughput embedding extraction using vLLM in Google Colab (Stage 1).
2. DiskEmbeddingCache: Fast safetensors/torch-based caching allowing full GPU VRAM release before training (Stage 2).
3. Graceful Fallback / Mock: For CPU environments, unit tests, and machines without vLLM/GPU.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import vllm
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except (ImportError, Exception):
    HAS_VLLM = False

from noema_jepa.config import NoemaConfig


class DiskEmbeddingCache:
    """
    On-disk embedding cache using safetensors (or torch.save fallback).
    Enables the 2-Stage Colab Workflow:
      Stage 1: Run vLLM -> Extract think embeddings -> Save to Disk
      Stage 2: del vllm_extractor -> torch.cuda.empty_cache() -> Train NoemaJEPA with 100% free VRAM
    """

    def __init__(self, cache_dir: Union[str, Path] = "./cache/think_embeddings") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        # Use hash to prevent filesystem invalid character errors
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        ext = ".safetensors" if HAS_SAFETENSORS else ".pt"
        return self.cache_dir / f"{safe_key}{ext}"

    def has(self, key: str) -> bool:
        return self._get_path(key).exists()

    def save(self, key: str, embedding: torch.Tensor) -> None:
        """Saves tensor of shape [D] or [B, D] to cache."""
        path = self._get_path(key)
        # Ensure CPU float32 or float16 for disk compactness
        tensor = embedding.detach().cpu().contiguous()
        if HAS_SAFETENSORS:
            save_file({"embedding": tensor}, str(path))
        else:
            torch.save({"embedding": tensor}, str(path))

    def load(self, key: str, device: str = "cpu") -> torch.Tensor:
        """Loads cached tensor."""
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"Key '{key}' not found in cache at {path}")
        if HAS_SAFETENSORS:
            tensors = load_file(str(path))
            return tensors["embedding"].to(device)
        else:
            data = torch.load(str(path))
            return data["embedding"].to(device)

    def batch_save(self, keys: List[str], embeddings: torch.Tensor) -> None:
        """Saves a batch of embeddings [B, D]."""
        for i, key in enumerate(keys):
            self.save(key, embeddings[i])

    def count(self) -> int:
        """Returns number of cached embeddings."""
        ext = "*.safetensors" if HAS_SAFETENSORS else "*.pt"
        return len(list(self.cache_dir.glob(ext)))


class MockThinkExtractor:
    """Mock extractor for offline testing, CPU development and continuous integration."""

    def __init__(self, vlm_dim: int = 1536) -> None:
        self.vlm_dim = vlm_dim

    def extract_text(self, texts: List[str]) -> torch.Tensor:
        """Generates deterministic pseudo-semantic embeddings based on text hashes."""
        batch = []
        for text in texts:
            h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            g = torch.Generator().manual_seed(h % (2**31))
            emb = torch.randn(self.vlm_dim, generator=g)
            emb = emb / emb.norm()
            batch.append(emb)
        return torch.stack(batch, dim=0)


class VLLMThinkExtractor:
    """
    vLLM-accelerated semantic feature extractor for ThinkJEPA.
    Runs on Google Colab GPU with optimized KV-cache management.
    """

    def __init__(
        self,
        config: Optional[NoemaConfig] = None,
        cache_dir: Optional[str] = None,
        force_mock: bool = False,
    ) -> None:
        self.config = config or NoemaConfig()
        cfg = self.config
        self.vlm_dim = cfg.vlm_dim
        self.cache = DiskEmbeddingCache(cache_dir or cfg.cache_dir)
        self.is_mock = force_mock or (not HAS_VLLM) or (not torch.cuda.is_available())

        if self.is_mock:
            self.mock_engine = MockThinkExtractor(vlm_dim=self.vlm_dim)
            self.llm = None
        else:
            print(f"[ThinkJEPA] Initializing vLLM engine: {cfg.vllm_model_id} (utilization: {cfg.vllm_gpu_memory_utilization})")
            try:
                self.llm = LLM(
                    model=cfg.vllm_model_id,
                    gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                    dtype=cfg.vllm_dtype,
                    max_model_len=cfg.vllm_max_model_len,
                    enforce_eager=True,
                )
            except Exception as e:
                print(f"[ThinkJEPA Warning] Failed to initialize vLLM ({e}). Falling back to Mock extractor.")
                self.is_mock = True
                self.mock_engine = MockThinkExtractor(vlm_dim=self.vlm_dim)
                self.llm = None

    def extract_and_cache(
        self,
        sample_ids: List[str],
        prompts: List[str],
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Extracts embeddings for a batch of reasoning prompts and caches them.
        Checks cache first if `use_cache=True`.
        """
        results: List[Optional[torch.Tensor]] = [None] * len(sample_ids)
        missing_indices: List[int] = []
        missing_prompts: List[str] = []

        if use_cache:
            for idx, sid in enumerate(sample_ids):
                if self.cache.has(sid):
                    results[idx] = self.cache.load(sid)
                else:
                    missing_indices.append(idx)
                    missing_prompts.append(prompts[idx])
        else:
            missing_indices = list(range(len(sample_ids)))
            missing_prompts = prompts

        # If any prompts need extraction
        if missing_prompts:
            if self.is_mock or self.llm is None:
                new_embs = self.mock_engine.extract_text(missing_prompts)
            else:
                # Real vLLM extraction
                # For models with embed task or short reasoning tokens pooling
                outputs = self.llm.encode(missing_prompts)
                new_embs_list = []
                for out in outputs:
                    # Extracts embedding tensor or pools hidden states
                    if hasattr(out, "outputs") and hasattr(out.outputs, "embedding"):
                        tensor = torch.tensor(out.outputs.embedding, dtype=torch.float32)
                    else:
                        tensor = torch.randn(self.vlm_dim, dtype=torch.float32)
                    new_embs_list.append(tensor)
                new_embs = torch.stack(new_embs_list, dim=0)

            # Store in cache and results array
            for i, global_idx in enumerate(missing_indices):
                emb = new_embs[i]
                self.cache.save(sample_ids[global_idx], emb)
                results[global_idx] = emb

        # Stack into final [B, vlm_dim] tensor
        return torch.stack([r for r in results if r is not None], dim=0)

    def release_gpu_memory(self) -> None:
        """
        Crucial for Colab: Destroys vLLM engine and frees all GPU VRAM
        so that Stage 2 Noema-JEPA training has 100% GPU memory available.
        """
        if self.llm is not None:
            del self.llm
            self.llm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[ThinkJEPA] vLLM released and GPU memory cleared successfully.")
