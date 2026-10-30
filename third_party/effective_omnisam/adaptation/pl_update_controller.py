from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PLUpdateDecision:
    should_update: bool
    reuse_epoch: int
    reason: str


class PseudoLabelUpdateController:
    """
    Minimal pseudo-label update controller.
    Current implemented modes:
    - every_epoch
    - every_n_epoch
    - score_based (fallback to every_epoch for now; interface reserved)
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "every_epoch",
        interval: int = 1,
        score_threshold: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = (mode or "every_epoch").lower()
        self.interval = max(1, int(interval))
        self.score_threshold = float(score_threshold)

    def decide(self, epoch: int) -> PLUpdateDecision:
        # epoch is 0-based
        if epoch <= 0:
            return PLUpdateDecision(True, epoch, "epoch0_always_update")
        if not self.enabled:
            return PLUpdateDecision(True, epoch, "controller_disabled")

        if self.mode == "every_epoch":
            return PLUpdateDecision(True, epoch, "every_epoch")

        if self.mode == "every_n_epoch":
            if epoch % self.interval == 0:
                return PLUpdateDecision(True, epoch, f"epoch_multiple_of_{self.interval}")
            reuse_epoch = epoch - (epoch % self.interval)
            return PLUpdateDecision(False, reuse_epoch, f"reuse_epoch{reuse_epoch + 1}_interval_{self.interval}")

        # Reserved for future score-based implementation
        if self.mode == "score_based":
            return PLUpdateDecision(True, epoch, "score_based_fallback_every_epoch")

        return PLUpdateDecision(True, epoch, "unknown_mode_fallback_every_epoch")
