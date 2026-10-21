"""
Noema-JEPA Data Module.
Provides traditional benchmark dataset adapters, synthetic multi-object generators,
and high-throughput dataloaders integrated with ThinkJEPA DiskEmbeddingCache.
"""

from noema_jepa.data.datasets import (
    BaseNoemaDataset,
    CLEVRDataset,
    CLEVRERDataset,
    SomethingSomethingV2Dataset,
    MultiObjectScenesDataset,
    KubricMOViDataset,
    ClassicVisionDataset,
    SyntheticToyDataset,
    ProceduralCLEVRERVideoDataset,
    ProceduralKubricVideoDataset,
    download_clevr_dataset,
    download_clevrer_dataset,
    download_kubric_dataset,
    read_video_frames,
    noema_collate_fn,
    build_noema_dataloader,
)

__all__ = [
    "BaseNoemaDataset",
    "CLEVRDataset",
    "CLEVRERDataset",
    "SomethingSomethingV2Dataset",
    "MultiObjectScenesDataset",
    "KubricMOViDataset",
    "ClassicVisionDataset",
    "SyntheticToyDataset",
    "ProceduralCLEVRERVideoDataset",
    "ProceduralKubricVideoDataset",
    "download_clevr_dataset",
    "download_clevrer_dataset",
    "download_kubric_dataset",
    "read_video_frames",
    "noema_collate_fn",
    "build_noema_dataloader",
]
