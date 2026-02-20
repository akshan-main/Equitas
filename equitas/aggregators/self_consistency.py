"""Method 3: Self-consistency (same LLM sampled K times, majority vote)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class SelfConsistencyAggregator(BaseAggregator):
    """
    Uses K samples from the same LLM, majority vote over them.
    The simulation engine generates K samples; this aggregator votes.
    """

    mechanism_meta = MechanismMeta(
        assumption="Errors across K samples are independent (temperature > 0)",
        works_when=[
            "LLM errors are stochastic, not systematic",
            "Higher temperature yields diverse but honest samples",
        ],
        breaks_when=[
            "Systematic model bias (same wrong answer K times)",
            "Corruption is at the model level, not agent level",
            "K is too small to outvote consistent errors",
        ],
        is_online=False,
    )

    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)
        self._weights = np.ones(num_samples) / num_samples

    def select(self, action_ids: List[str], round_id: int) -> str:
        return weighted_vote(action_ids, self._weights)

    def update(self, losses: List[float], round_id: int) -> None:
        pass

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def name(self) -> str:
        return "self_consistency"
