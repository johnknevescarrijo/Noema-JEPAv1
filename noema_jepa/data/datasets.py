"""
Traditional Benchmark Datasets and DataLoaders for Noema-JEPA v1.

Includes:
1. CLEVR / CLEVRER: Object-centric slot attention & causal physical reasoning.
2. Something-Something V2 (SSv2): Temporal world models & physical action dynamics.
3. MS-COCO / PASCAL VOC: Multi-object discovery in unconstrained real-world scenes.
4. Kubric MOVi (MOVi-A / MOVi-C) / PHYRE: Rigid-body physical dynamics & video slots.
5. Classic Vision (CIFAR-100, STL-10, ImageNet-100): Fast pre-training and linear probing.
6. SyntheticToyDataset: Zero-download procedural multi-object generator for instant Colab testing.
7. DiskEmbeddingCache integration: Seamless coupling with ThinkJEPA vLLM representations.
8. build_noema_dataloader: Universal factory function for unified training and evaluation.
"""

import os
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

from noema_jepa.think_branch.vllm_extractor import DiskEmbeddingCache


# ---------------------------------------------------------------------------
# Default Transformations
# ---------------------------------------------------------------------------

def get_default_transforms(image_size: int = 128, is_train: bool = True) -> transforms.Compose:
    """Returns standardized image transformations for Noema-JEPA."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Base Dataset Class
# ---------------------------------------------------------------------------

class BaseNoemaDataset(Dataset):
    """
    Abstract base dataset providing unified caching and return formats.
    Each item yields:
      {
          "images": torch.Tensor [C, H, W] or [T, C, H, W],
          "think_embeddings": Optional[torch.Tensor] [vlm_dim] (if cached),
          "sample_id": str,
          "vlm_prompt": str
      }
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        num_frames: int = 1,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.transform = transform or get_default_transforms(image_size, is_train=(split == "train"))
        self.max_samples = max_samples

        if cache is not None:
            self.cache = cache
        elif cache_dir is not None:
            self.cache = DiskEmbeddingCache(cache_dir)
        else:
            self.cache = None

    def _get_think_embedding(self, sample_id: str) -> Optional[torch.Tensor]:
        """Safely queries DiskEmbeddingCache for pre-extracted ThinkJEPA embeddings."""
        if self.cache is not None and self.cache.has(sample_id):
            try:
                return self.cache.load(sample_id)
            except Exception:
                return None
        return None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Procedural Multi-Object Synthetic Generator (Instant Colab Fallback)
# ---------------------------------------------------------------------------

class SyntheticToyDataset(BaseNoemaDataset):
    """
    Procedural dataset generating multi-object scenes on the fly.
    Allows running the full Noema-JEPA pipeline with zero download latency.
    Supports both static images [C, H, W] and dynamic video clips [T, C, H, W].
    """

    COLORS = [
        (220, 50, 47),   # Red
        (38, 139, 210),  # Blue
        (133, 153, 0),   # Green
        (181, 137, 0),   # Yellow
        (211, 54, 130),  # Magenta
        (42, 161, 152),  # Cyan
        (203, 75, 22),   # Orange
        (108, 113, 196), # Violet
    ]

    SHAPES = ["circle", "square", "triangle"]

    def __init__(
        self,
        num_samples: int = 1000,
        split: str = "train",
        image_size: int = 128,
        num_frames: int = 1,
        num_objects: int = 5,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            cache=cache,
            cache_dir=cache_dir,
        )
        self.num_samples = num_samples
        self.num_objects = num_objects
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _render_frame(self, objects: List[Dict[str, Any]], t: int) -> Image.Image:
        """Renders 2D geometric objects with bouncing physics onto PIL Canvas."""
        img = Image.new("RGB", (self.image_size, self.image_size), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)

        for obj in objects:
            # Simple linear kinematic update with boundary reflection
            w = self.image_size - obj["size"]
            x = (obj["x0"] + obj["vx"] * t) % (2 * w)
            if x > w:
                x = 2 * w - x

            y = (obj["y0"] + obj["vy"] * t) % (2 * w)
            if y > w:
                y = 2 * w - y

            s = obj["size"]
            color = obj["color"]
            shape = obj["shape"]

            if shape == "circle":
                draw.ellipse([x, y, x + s, y + s], fill=color, outline=(20, 20, 20))
            elif shape == "square":
                draw.rectangle([x, y, x + s, y + s], fill=color, outline=(20, 20, 20))
            elif shape == "triangle":
                p1 = (x + s // 2, y)
                p2 = (x, y + s)
                p3 = (x + s, y + s)
                draw.polygon([p1, p2, p3], fill=color, outline=(20, 20, 20))

        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deterministic generation per sample index
        rng = random.Random(self.seed + idx * 1000)
        objects = []
        for _ in range(self.num_objects):
            objects.append({
                "x0": rng.randint(0, self.image_size // 2),
                "y0": rng.randint(0, self.image_size // 2),
                "vx": rng.randint(2, 6),
                "vy": rng.randint(2, 6),
                "size": rng.randint(18, 32),
                "shape": rng.choice(self.SHAPES),
                "color": rng.choice(self.COLORS),
            })

        sample_id = f"synthetic_{self.split}_{idx:06d}"
        vlm_prompt = (
            f"Physical scene containing {self.num_objects} moving discrete geometric objects "
            f"under elastic collisions."
        )

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.num_frames <= 1:
            pil_img = self._render_frame(objects, t=0)
            img_tensor = to_tensor(pil_img)  # [C, H, W]
        else:
            frames = [to_tensor(self._render_frame(objects, t=t)) for t in range(self.num_frames)]
            img_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": img_tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# 2. CLEVR / CLEVRER Dataset Adapter (Object-Centric Reasoning)
# ---------------------------------------------------------------------------

class CLEVRDataset(BaseNoemaDataset):
    """
    CLEVR / CLEVRER: Classic benchmark for object-centric representation learning,
    slot attention discovery, and causal physical reasoning (Johnson et al., Yi et al.).
    
    Loads official CLEVR directories if present:
      - Images: `{root}/images/{split}/CLEVR_{split}_{id:06d}.png`
      - Scenes JSON: `{root}/scenes/CLEVR_{split}_scenes.json`
    Falls back gracefully to high-fidelity procedural generation if raw files are absent.
    """

    def __init__(
        self,
        root: Union[str, Path] = "./data/CLEVR_v1.0",
        split: str = "train",
        image_size: int = 128,
        num_frames: int = 1,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        synthetic_fallback: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        self.root = Path(root)
        self.synthetic_fallback = synthetic_fallback
        self.image_paths: List[Path] = []
        self.scene_metadata: Dict[str, Any] = {}

        # Look for local CLEVR structure
        img_dir = self.root / "images" / split
        if img_dir.exists():
            found = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
            if found:
                self.image_paths = found
                scenes_json = self.root / "scenes" / f"CLEVR_{split}_scenes.json"
                if scenes_json.exists():
                    try:
                        with open(scenes_json, "r") as f:
                            data = json.load(f)
                            self.scene_metadata = {s["image_filename"]: s for s in data.get("scenes", [])}
                    except Exception:
                        pass

        if not self.image_paths and self.synthetic_fallback:
            self._toy = SyntheticToyDataset(
                num_samples=max_samples or 1000,
                split=split,
                image_size=image_size,
                num_frames=num_frames,
                num_objects=6,
                cache=self.cache,
            )
        else:
            self._toy = None

        if self.max_samples and len(self.image_paths) > self.max_samples:
            self.image_paths = self.image_paths[:self.max_samples]

    def __len__(self) -> int:
        if self._toy is not None:
            return len(self._toy)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._toy is not None:
            return self._toy[idx]

        img_path = self.image_paths[idx]
        sample_id = f"clevr_{self.split}_{img_path.stem}"

        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(pil_img)

        # Build prompt from metadata if available
        meta = self.scene_metadata.get(img_path.name, {})
        obj_count = len(meta.get("objects", []))
        if obj_count > 0:
            vlm_prompt = (
                f"CLEVR scene with {obj_count} 3D objects (spheres, cubes, cylinders). "
                f"Analyze spatial coordinates, colors, and occlusions."
            )
        else:
            vlm_prompt = "CLEVR scene with multiple 3D objects. Disentangle spatial object slots."

        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": img_tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# 3. Something-Something V2 Dataset Adapter (Temporal Dynamics)
# ---------------------------------------------------------------------------

class SomethingSomethingV2Dataset(BaseNoemaDataset):
    """
    Something-Something V2 (SSv2): Traditional benchmark for temporal world modeling
    and causal human-object interactions (Goyal et al., 2017).
    
    Reads frame sequences or video directories.
    Provides temporal video representations [T, C, H, W] for C-JEPA prediction.
    """

    def __init__(
        self,
        root: Union[str, Path] = "./data/ssv2",
        split: str = "train",
        image_size: int = 128,
        num_frames: int = 4,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        synthetic_fallback: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        self.root = Path(root)
        self.synthetic_fallback = synthetic_fallback
        self.video_dirs: List[Path] = []

        frames_dir = self.root / "frames" / split
        if frames_dir.exists():
            dirs = [d for d in sorted(frames_dir.iterdir()) if d.is_dir()]
            if dirs:
                self.video_dirs = dirs

        if not self.video_dirs and self.synthetic_fallback:
            self._toy = SyntheticToyDataset(
                num_samples=max_samples or 1000,
                split=split,
                image_size=image_size,
                num_frames=num_frames,
                num_objects=4,
                cache=self.cache,
            )
        else:
            self._toy = None

        if self.max_samples and len(self.video_dirs) > self.max_samples:
            self.video_dirs = self.video_dirs[:self.max_samples]

    def __len__(self) -> int:
        if self._toy is not None:
            return len(self._toy)
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._toy is not None:
            return self._toy[idx]

        vdir = self.video_dirs[idx]
        sample_id = f"ssv2_{self.split}_{vdir.name}"
        frame_files = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))

        if len(frame_files) == 0:
            # Fallback to single blank image if directory empty
            tensor = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        else:
            # Uniformly subsample num_frames
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            sampled_frames = []
            for i in indices:
                img = Image.open(frame_files[i]).convert("RGB")
                sampled_frames.append(self.transform(img))
            tensor = torch.stack(sampled_frames, dim=0)

        vlm_prompt = (
            f"Something-Something V2 action video clip ID {vdir.name}. "
            f"Reason about physical object motion, hand interaction, and target state."
        )

        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# 4. MS-COCO / PASCAL VOC Dataset Adapter (Multi-Object Real Scenes)
# ---------------------------------------------------------------------------

class MultiObjectScenesDataset(BaseNoemaDataset):
    """
    MS-COCO & PASCAL VOC: Classic multi-object real-world scene benchmarks
    (Lin et al., Everingham et al.).
    
    Tests whether Slot Attention and C-JEPA discover discrete physical object
    boundaries under complex natural textures, severe occlusions, and varied lighting.
    """

    def __init__(
        self,
        root: Union[str, Path] = "./data/coco",
        dataset_type: str = "coco",  # "coco" or "voc"
        split: str = "train",
        image_size: int = 128,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        synthetic_fallback: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=1,
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        self.root = Path(root)
        self.dataset_type = dataset_type.lower()
        self.synthetic_fallback = synthetic_fallback
        self.image_paths: List[Path] = []

        # Check standard paths
        if self.dataset_type == "coco":
            img_dir = self.root / f"{split}2017"
            if not img_dir.exists():
                img_dir = self.root / split
            if img_dir.exists():
                self.image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        else:  # VOC
            voc_dir = self.root / "VOCdevkit" / "VOC2012" / "JPEGImages"
            if voc_dir.exists():
                self.image_paths = sorted(list(voc_dir.glob("*.jpg")))

        if not self.image_paths and self.synthetic_fallback:
            self._toy = SyntheticToyDataset(
                num_samples=max_samples or 1000,
                split=split,
                image_size=image_size,
                num_frames=1,
                num_objects=7,
                cache=self.cache,
            )
        else:
            self._toy = None

        if self.max_samples and len(self.image_paths) > self.max_samples:
            self.image_paths = self.image_paths[:self.max_samples]

    def __len__(self) -> int:
        if self._toy is not None:
            return len(self._toy)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._toy is not None:
            return self._toy[idx]

        img_path = self.image_paths[idx]
        sample_id = f"{self.dataset_type}_{self.split}_{img_path.stem}"

        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(pil_img)

        vlm_prompt = (
            f"Real-world multi-object complex scene from {self.dataset_type.upper()}. "
            f"Segment discrete foreground physical objects and spatial context."
        )

        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": img_tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# 5. Kubric MOVi / PHYRE Dataset Adapter (Intuitive Physics World Models)
# ---------------------------------------------------------------------------

class KubricMOViDataset(BaseNoemaDataset):
    """
    Kubric MOVi (MOVi-A / MOVi-C / MOVi-E) and PHYRE:
    Traditional benchmarks for intuitive physics, multi-object rigid-body dynamics,
    and video slot attention (Greff et al., 2022; Bakhtin et al., 2019).
    """

    def __init__(
        self,
        root: Union[str, Path] = "./data/kubric",
        movi_split: str = "movi_c",
        split: str = "train",
        image_size: int = 128,
        num_frames: int = 4,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        synthetic_fallback: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        self.root = Path(root)
        self.movi_split = movi_split
        self.synthetic_fallback = synthetic_fallback
        self.scene_dirs: List[Path] = []

        sub_dir = self.root / movi_split / split
        if sub_dir.exists():
            self.scene_dirs = [d for d in sorted(sub_dir.iterdir()) if d.is_dir()]

        if not self.scene_dirs and self.synthetic_fallback:
            self._toy = SyntheticToyDataset(
                num_samples=max_samples or 1000,
                split=split,
                image_size=image_size,
                num_frames=num_frames,
                num_objects=5,
                cache=self.cache,
            )
        else:
            self._toy = None

        if self.max_samples and len(self.scene_dirs) > self.max_samples:
            self.scene_dirs = self.scene_dirs[:self.max_samples]

    def __len__(self) -> int:
        if self._toy is not None:
            return len(self._toy)
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._toy is not None:
            return self._toy[idx]

        sdir = self.scene_dirs[idx]
        sample_id = f"kubric_{self.movi_split}_{self.split}_{sdir.name}"
        frame_files = sorted(list(sdir.glob("rgba_*.png")) + list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))

        if len(frame_files) == 0:
            tensor = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        else:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            sampled_frames = []
            for i in indices:
                img = Image.open(frame_files[i]).convert("RGB")
                sampled_frames.append(self.transform(img))
            tensor = torch.stack(sampled_frames, dim=0)

        vlm_prompt = (
            f"Kubric {self.movi_split.upper()} physical simulation clip {sdir.name}. "
            f"Predict physical trajectories and collision dynamics among interacting bodies."
        )

        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# 6. Classic Vision Benchmarks (CIFAR-100, STL-10, ImageNet-100)
# ---------------------------------------------------------------------------

class ClassicVisionDataset(BaseNoemaDataset):
    """
    Standard Vision Benchmarks for rapid pretraining in Google Colab:
    - CIFAR-100 (Krizhevsky, 2009)
    - STL-10 (Coates et al., 2011)
    - ImageNet-100 (Deng et al., 2009 subset)
    
    Equipped with automated torchvision download and uniform resizing to img_size.
    """

    def __init__(
        self,
        name: str = "stl10",
        root: Union[str, Path] = "./data",
        split: str = "train",
        image_size: int = 128,
        download: bool = False,
        transform: Optional[Callable] = None,
        cache: Optional[DiskEmbeddingCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(
            split=split,
            image_size=image_size,
            num_frames=1,
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        self.name = name.lower()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        is_train = split in ("train", "train+unlabeled")

        if self.name in ("cifar100", "cifar"):
            from torchvision.datasets import CIFAR100
            self._ds = CIFAR100(root=str(self.root), train=is_train, download=download, transform=self.transform)
        elif self.name == "stl10":
            from torchvision.datasets import STL10
            stl_split = "train+unlabeled" if is_train else "test"
            self._ds = STL10(root=str(self.root), split=stl_split, download=download, transform=self.transform)
        elif self.name in ("imagenet100", "imagenet"):
            from torchvision.datasets import ImageFolder
            target_dir = self.root / self.name / split
            if target_dir.exists():
                self._ds = ImageFolder(root=str(target_dir), transform=self.transform)
            else:
                # Fallback to synthetic if ImageNet folder is not present
                self._ds = SyntheticToyDataset(num_samples=1000, split=split, image_size=image_size, cache=self.cache)
        else:
            raise ValueError(f"Unknown classic vision dataset: {name}")

        self._length = len(self._ds)
        if self.max_samples and self._length > self.max_samples:
            self._length = self.max_samples

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._ds[idx]
        sample_id = f"{self.name}_{self.split}_{idx:06d}"

        if isinstance(item, dict):
            # Already a dict from SyntheticToyDataset
            return item

        img_tensor = item[0]
        vlm_prompt = f"Natural image from {self.name.upper()} dataset. Extract high-level visual concept."
        think_emb = self._get_think_embedding(sample_id)

        return {
            "images": img_tensor,
            "think_embeddings": think_emb,
            "sample_id": sample_id,
            "vlm_prompt": vlm_prompt,
        }


# ---------------------------------------------------------------------------
# Collate Function & Factory
# ---------------------------------------------------------------------------

def noema_collate_fn(batch: List[Union[Dict[str, Any], Tuple[Any, ...]]]) -> Dict[str, Any]:
    """
    Collate function ensuring robust tensor assembly for ColabTrainer.
    Handles:
      - Images [B, C, H, W] or [B, T, C, H, W]
      - Think embeddings [B, D] (or None if unpopulated)
      - Sample IDs and Prompts lists
    """
    if isinstance(batch[0], dict):
        images_list = [item["images"] for item in batch]
        sample_ids = [item.get("sample_id", f"sample_{i}") for i, item in enumerate(batch)]
        prompts = [item.get("vlm_prompt", "") for item in batch]

        # Stack images
        images = torch.stack(images_list, dim=0)

        # Check think embeddings
        think_list = [item.get("think_embeddings") for item in batch]
        if all(t is not None and isinstance(t, torch.Tensor) for t in think_list):
            think_embeddings = torch.stack(think_list, dim=0)
        else:
            think_embeddings = None

        return {
            "images": images,
            "think_embeddings": think_embeddings,
            "sample_ids": sample_ids,
            "vlm_prompts": prompts,
        }
    elif isinstance(batch[0], (list, tuple)):
        images = torch.stack([item[0] for item in batch], dim=0)
        if len(batch[0]) > 1 and isinstance(batch[0][1], torch.Tensor):
            think_embeddings = torch.stack([item[1] for item in batch], dim=0)
        else:
            think_embeddings = None
        return {
            "images": images,
            "think_embeddings": think_embeddings,
        }
    else:
        return {"images": torch.stack(batch, dim=0), "think_embeddings": None}


def build_noema_dataloader(
    dataset_name: str,
    root: Union[str, Path] = "./data",
    split: str = "train",
    batch_size: int = 256,
    image_size: int = 128,
    num_frames: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    download: bool = False,
    synthetic_fallback: bool = True,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Universal factory function for building high-throughput Noema-JEPA DataLoaders.

    Supported benchmark identifiers:
      - 'clevr', 'clevrer'                  -> CLEVRDataset
      - 'ssv2', 'somethingsomethingv2'      -> SomethingSomethingV2Dataset
      - 'coco', 'voc', 'multiobject'        -> MultiObjectScenesDataset
      - 'kubric', 'movi', 'movi_a', 'movi_c'-> KubricMOViDataset
      - 'stl10', 'cifar100', 'imagenet100'  -> ClassicVisionDataset
      - 'synthetic', 'toy'                  -> SyntheticToyDataset

    Args:
        dataset_name: Target benchmark name (case-insensitive).
        root: Base storage path for raw datasets.
        split: Dataset split ('train', 'val', 'test').
        batch_size: Batch size for training or evaluation (default 256 for A100).
        image_size: Square image resolution (default 128x128).
        num_frames: Frames per clip (1 for image, >1 for video).
        num_workers: Subprocess data workers for DataLoader (default 4).
        shuffle: Whether to shuffle samples each epoch.
        cache_dir: Directory containing precomputed ThinkJEPA safetensors embeddings.
        download: Auto-download dataset if supported by torchvision.
        synthetic_fallback: If True, uses zero-download procedural generator if files are absent.
        pin_memory: Speed up CPU-to-GPU memory transfer on CUDA runtimes (auto-disabled on CPU).
        persistent_workers: Keep worker processes alive between epochs (defaults to True if num_workers > 0).
        prefetch_factor: Number of batches loaded in advance by each worker (default 2 if num_workers > 0).
        drop_last: Drop the last non-full batch to stabilize VICReg batch statistics.
        max_samples: Optional cap on dataset size for rapid testing.

    Returns:
        torch.utils.data.DataLoader yielding standardized Noema-JEPA batches.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")

    cache = DiskEmbeddingCache(cache_dir) if cache_dir is not None else None

    if "clevr" in key:
        dataset = CLEVRDataset(
            root=root,
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            cache=cache,
            max_samples=max_samples,
            synthetic_fallback=synthetic_fallback,
        )
    elif "ssv2" in key or "somethingsomething" in key:
        dataset = SomethingSomethingV2Dataset(
            root=root,
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            cache=cache,
            max_samples=max_samples,
            synthetic_fallback=synthetic_fallback,
        )
    elif "coco" in key or "voc" in key or "multiobject" in key:
        ds_type = "voc" if "voc" in key else "coco"
        dataset = MultiObjectScenesDataset(
            root=root,
            dataset_type=ds_type,
            split=split,
            image_size=image_size,
            cache=cache,
            max_samples=max_samples,
            synthetic_fallback=synthetic_fallback,
        )
    elif "kubric" in key or "movi" in key or "phyre" in key:
        movi_split = "movi_a" if "movia" in key else "movi_c"
        dataset = KubricMOViDataset(
            root=root,
            movi_split=movi_split,
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            cache=cache,
            max_samples=max_samples,
            synthetic_fallback=synthetic_fallback,
        )
    elif key in ("cifar100", "cifar", "stl10", "imagenet100", "imagenet"):
        dataset = ClassicVisionDataset(
            name=dataset_name,
            root=root,
            split=split,
            image_size=image_size,
            download=download,
            cache=cache,
            max_samples=max_samples,
        )
    elif key in ("synthetic", "toy"):
        dataset = SyntheticToyDataset(
            num_samples=max_samples or 1000,
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            cache=cache,
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from: ['clevr', 'clevrer', 'ssv2', 'coco', 'voc', 'kubric', 'movi', 'stl10', 'cifar100', 'synthetic']"
        )

    use_pin_memory = pin_memory and torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": noema_collate_fn,
        "pin_memory": use_pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True if persistent_workers is None else persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    elif persistent_workers is not None and persistent_workers:
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(**loader_kwargs)
    return loader
