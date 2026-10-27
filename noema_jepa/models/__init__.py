"""
Models package for Noema-JEPA v1.
"""

from noema_jepa.models.film import ZeroInitResidualFiLM
from noema_jepa.models.encoder import SpatialPatchEncoder, SlotAttention
from noema_jepa.models.quantizer import FSQSlotQuantizer
from noema_jepa.models.predictor import CJEPAObjectPredictor, CJEPABlock
from noema_jepa.models.noema_jepa import NoemaJEPA, ThinkJEPABridge

__all__ = [
    "ZeroInitResidualFiLM",
    "SpatialPatchEncoder",
    "SlotAttention",
    "FSQSlotQuantizer",
    "CJEPAObjectPredictor",
    "CJEPABlock",
    "NoemaJEPA",
    "ThinkJEPABridge",
]
