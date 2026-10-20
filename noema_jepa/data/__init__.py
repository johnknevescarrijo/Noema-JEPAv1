"""
Noema-JEPA Data Module.
Provides traditional benchmark dataset adapters, synthetic multi-object generators,
and high-throughput dataloaders integrated with ThinkJEPA DiskEmbeddingCache.
"""

from noema_jepa.data.datasets import (
    BaseNoemaDataset,
    CLEVRDataset,
    SomethingSomethingV2Dataset,
    MultiObjectScenesDataset,
    KubricMOViDataset,
    ClassicVisionDataset,
    SyntheticToyDataset,
    noema_collate_fn,
    build_noema_dataloader,
)

__all__ = [
    "BaseNoemaDataset",
    "CLEVRDataset",
    "SomethingSomethingV2Dataset",
    "MultiObjectScenesDataset",
    "KubricMOViDataset",
    "ClassicVisionDataset",
    "SyntheticToyDataset",
    "noema_collate_fn",
    "build_noema_dataloader",
]
