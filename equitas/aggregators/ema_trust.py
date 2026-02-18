"""Method 4: EMA trust (exponential moving average of agent accuracy)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class EMATrustAggregator(BaseAggregator):
    """w_i = ema_alpha * (1 - loss_i) + (1 - ema_alpha) * w_i, then normalize."""

    mechanism_meta = MechanismMeta(
        assumption="Recent performance predicts near-future performance",
        works_when=[
            "Corruption is stationary or slowly drifting",
            "Agent quality is persistent across rounds",
        ],
        breaks_when=[
            "Delayed-onset corruption (honest phase builds trust, then exploits)",
            "Rapid non-stationary shifts in corruption pattern",
            "Alpha too high forgets history; too low adapts slowly",
        ],
        is_online=True,
    )

    def __init__(self, num_agents: int, ema_alpha: float = 0.3) -> None:
        super().__init__(num_agents)
        self.ema_alpha = ema_alpha
        self._weights = np.ones(num_agents) / num_agents

    def select(self, action_ids: List[str], round_id: int) -> str:
        return weighted_vote(action_ids, self._weights)

    def update(self, losses: List[float], round_id: int) -> None:
        loss_arr = np.array(losses)
        accuracy = 1.0 - np.clip(loss_arr, 0.0, 1.0)
        self._weights = self.ema_alpha * accuracy + (1.0 - self.ema_alpha) * self._weights
        total = self._weights.sum()
        if total > 0:
            self._weights /= total
        else:
            self._weights = np.ones(self.num_agents) / self.num_agents

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def reset(self) -> None:
        self._weights = np.ones(self.num_agents) / self.num_agents

    @property
    def name(self) -> str:
        return "ema_trust"
