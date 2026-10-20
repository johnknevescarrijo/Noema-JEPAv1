"""
Variance Watchdog for Noema-JEPA v1.

Real-time monitoring system designed to detect and prevent representation collapse
and FiLM overshadowing in contextually modulated JEPA architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from noema_jepa.config import NoemaConfig


@dataclass
class WatchdogReport:
    """Diagnostic report generated at each step by the VarianceWatchdog."""
    step: int
    status: str  # "HEALTHY", "WARNING", "CRITICAL"
    batch_std: float
    intra_scene_std: float
    film_gate_norm: float
    current_modulation_scale: float
    current_think_weight: float
    action_taken: str


class VarianceWatchdog:
    """
    Monitors latent variance in real time during training:
    - batch_std: Variance across samples in batch (VICReg metric)
    - intra_scene_std: Variance across object slots within the same scene
    - film_gate_norm: Norm of FiLM gate alpha

    If intra_scene_std or batch_std drops below std_threshold:
    Automatically damps ThinkJEPA loss weight and FiLM modulation scale to prevent
    the global VLM reasoning from flattening distinct object slots.
    """

    def __init__(self, config: Optional[NoemaConfig] = None) -> None:
        self.config = config or NoemaConfig()
        cfg = self.config

        self.threshold = cfg.watchdog_std_threshold
        self.recovery_threshold = cfg.watchdog_std_threshold * cfg.watchdog_recovery_factor
        self.decay_rate = cfg.watchdog_decay_rate
        self.recovery_rate = cfg.watchdog_recovery_rate
        self.min_scale = cfg.min_film_modulation_scale
        self.max_scale = cfg.max_film_modulation_scale

        self.nominal_think_weight = cfg.think_loss_weight
        self.current_think_weight = cfg.think_loss_weight
        self.current_modulation_scale = cfg.initial_film_modulation_scale

        self.step_count = 0
        self.history: List[WatchdogReport] = []
        self.collapse_warnings_total = 0

    def inspect_film_gates(self, model: nn.Module) -> float:
        """Inspects average alpha magnitude across all FiLM layers in the model."""
        alphas = []
        for mod in model.modules():
            if hasattr(mod, "alpha") and isinstance(mod.alpha, nn.Parameter):
                alphas.append(mod.alpha.detach().abs().mean().item())

        if not alphas:
            return 0.0
        return sum(alphas) / len(alphas)

    def step(
        self,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> WatchdogReport:
        """
        Executes one watchdog verification step.

        Args:
            metrics: Metrics dict from NoemaWorldModelLoss (containing std values).
            model: Optional NoemaJEPA model instance.
            loss_fn: Optional NoemaWorldModelLoss instance to apply weight damping.

        Returns:
            WatchdogReport with status and updated scaling factors.
        """
        self.step_count += 1

        batch_std = metrics.get("vicreg_std_pred", 1.0)
        intra_scene_std = metrics.get("metric_intra_scene_std", 1.0)

        film_gate_norm = (
            self.inspect_film_gates(model) if model is not None else 0.05
        )

        # Check for representational collapse
        is_critical = (batch_std < self.threshold * 0.5) or (intra_scene_std < self.threshold * 0.5)
        is_warning = (batch_std < self.threshold) or (intra_scene_std < self.threshold)
        is_healthy = (batch_std > self.recovery_threshold) and (intra_scene_std > self.recovery_threshold)

        action = "None"
        if is_critical or is_warning:
            self.collapse_warnings_total += 1
            status = "CRITICAL" if is_critical else "WARNING"

            # Apply aggressive damping
            rate = self.decay_rate if is_warning else (self.decay_rate ** 1.5)
            self.current_modulation_scale = max(
                self.min_scale,
                self.current_modulation_scale * rate,
            )
            self.current_think_weight = max(
                1e-4,
                self.current_think_weight * rate,
            )
            action = f"Damped FiLM scale -> {self.current_modulation_scale:.3f}, Think weight -> {self.current_think_weight:.4f}"

        elif is_healthy:
            status = "HEALTHY"
            # Gradually restore towards nominal scale
            if self.current_modulation_scale < self.max_scale:
                self.current_modulation_scale = min(
                    self.max_scale,
                    self.current_modulation_scale * self.recovery_rate,
                )
                action = f"Recovering FiLM scale -> {self.current_modulation_scale:.3f}"

            if self.current_think_weight < self.nominal_think_weight:
                self.current_think_weight = min(
                    self.nominal_think_weight,
                    self.current_think_weight * self.recovery_rate,
                )
                action += f", Think weight -> {self.current_think_weight:.4f}"
        else:
            status = "STABLE"

        # Apply updated think weight to loss function if provided
        if loss_fn is not None and hasattr(loss_fn, "update_think_loss_weight"):
            loss_fn.update_think_loss_weight(self.current_think_weight)

        report = WatchdogReport(
            step=self.step_count,
            status=status,
            batch_std=batch_std,
            intra_scene_std=intra_scene_std,
            film_gate_norm=film_gate_norm,
            current_modulation_scale=self.current_modulation_scale,
            current_think_weight=self.current_think_weight,
            action_taken=action,
        )

        self.history.append(report)
        return report

    def get_modulation_scale(self) -> float:
        """Returns the active FiLM modulation scale to pass into forward()."""
        return self.current_modulation_scale

    def get_summary(self) -> Dict[str, float]:
        """Summary of watchdog health history."""
        return {
            "total_steps": self.step_count,
            "collapse_warnings_total": self.collapse_warnings_total,
            "current_modulation_scale": self.current_modulation_scale,
            "current_think_weight": self.current_think_weight,
        }
