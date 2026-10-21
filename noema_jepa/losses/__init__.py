"""
Loss functions package for Noema-JEPA v1.
"""

from noema_jepa.losses.vicreg import VICRegLoss, NoemaWorldModelLoss, compute_loss

__all__ = ["VICRegLoss", "NoemaWorldModelLoss", "compute_loss"]
