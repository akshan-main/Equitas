"""Method 6: Multiplicative Weights (online learning aggregator)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class MultiplicativeWeightsAggregator(BaseAggregator):
    """w_i <- w_i * exp(-eta * loss_i), then renormalize."""

    mechanism_meta = MechanismMeta(
        assumption="Per-agent losses are observable after each round",
        works_when=[
            "Exponential down-weighting should isolate consistently high-loss agents",
            "Expected to recover after corruption onset due to multiplicative updates",
            "Theory predicts finite regret for any ε < 1 (if loss signal is clean)",
        ],
        breaks_when=[
            "Loss signal is corrupted (e.g., deceptive adversaries fool judges)",
            "Very short horizons (T < 10): insufficient rounds for weight separation",
            "η too high overshoots; η too low adapts slowly",
        ],
        is_online=True,
    )

    def __init__(
        self,
        num_agents: int,
        eta: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        min_weight: float = 1e-6,
    ) -> None:
        super().__init__(num_agents)
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.min_weight = min_weight
        self._weights = np.ones(num_agents) / num_agents

    def select(self, action_ids: List[str], round_id: int) -> str:
        return weighted_vote(action_ids, self._weights)

    def update(self, losses: List[float], round_id: int) -> None:
        loss_arr = np.array(losses)
        self._weights *= np.exp(-self.eta * loss_arr)
        self._weights = np.maximum(self._weights, self.min_weight)
        total = self._weights.sum()
        if total > 0:
            self._weights /= total

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def reset(self) -> None:
        self._weights = np.ones(self.num_agents) / self.num_agents

    @property
    def name(self) -> str:
        return "multiplicative_weights"
