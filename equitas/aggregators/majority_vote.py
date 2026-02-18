"""Method 1: Majority vote (equal-weight plurality)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class MajorityVoteAggregator(BaseAggregator):

    mechanism_meta = MechanismMeta(
        assumption="Majority of agents are honest (ε < 0.5)",
        works_when=[
            "Random noise with independent errors",
            "Low corruption rate (ε < 0.5)",
            "No coordination among corrupted agents",
        ],
        breaks_when=[
            "Coordinated corruption concentrates votes on one wrong action",
            "Corruption rate ε ≥ 0.5",
            "Systematic bias is consistent across corrupted agents",
        ],
        is_online=False,
    )

    def __init__(self, num_agents: int) -> None:
        super().__init__(num_agents)
        self._weights = np.ones(num_agents) / num_agents

    def select(self, action_ids: List[str], round_id: int) -> str:
        return weighted_vote(action_ids, self._weights)

    def update(self, losses: List[float], round_id: int) -> None:
        pass

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def name(self) -> str:
        return "majority_vote"
