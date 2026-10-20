"""
Noema-JEPA v1: Contextual Slot-based Joint Embedding Predictive Architecture (~15M params)
with ThinkJEPA vLLM Bridge and Variance Watchdog.
"""

from noema_jepa.config import NoemaConfig
from noema_jepa.models.noema_jepa import NoemaJEPA, ThinkJEPABridge
from noema_jepa.models.encoder import SpatialPatchEncoder, SlotAttention
from noema_jepa.models.predictor import CJEPAObjectPredictor, CJEPABlock
from noema_jepa.models.film import ZeroInitResidualFiLM
from noema_jepa.losses.vicreg import VICRegLoss, NoemaWorldModelLoss
from noema_jepa.monitoring.variance_watchdog import VarianceWatchdog, WatchdogReport
from noema_jepa.think_branch.vllm_extractor import VLLMThinkExtractor, DiskEmbeddingCache
from noema_jepa.train.colab_trainer import ColabTrainer
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

__version__ = "0.1.0"

__all__ = [
    "NoemaConfig",
    "NoemaJEPA",
    "ThinkJEPABridge",
    "SpatialPatchEncoder",
    "SlotAttention",
    "CJEPAObjectPredictor",
    "CJEPABlock",
    "ZeroInitResidualFiLM",
    "VICRegLoss",
    "NoemaWorldModelLoss",
    "VarianceWatchdog",
    "WatchdogReport",
    "VLLMThinkExtractor",
    "DiskEmbeddingCache",
    "ColabTrainer",
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
