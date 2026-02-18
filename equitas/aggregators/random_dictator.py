"""Method 8: Random dictator (uniform random agent selection per round)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta


class RandomDictatorAggregator(BaseAggregator):
    """Each round, pick one agent uniformly at random and use their action.

    Trivial baseline. Expected accuracy under ε-corruption:
        P(honest dictator) = (1-ε)

    No learning, no state. Included as a lower bound that every
    social choice benchmark should report.
    """

    mechanism_meta = MechanismMeta(
        assumption="None — requires no assumptions about agent quality",
        works_when=[
            "Low corruption rate: P(honest) = 1-ε per round",
            "As a calibration lower bound — any useful mechanism should beat this",
        ],
        breaks_when=[
            "Any corruption rate > 0 (expected regret scales linearly with ε)",
            "Single bad draw can have outsized impact",
        ],
        is_online=False,
    )

    def __init__(self, num_agents: int, seed: int = 0) -> None:
        super().__init__(num_agents)
        self._rng = np.random.default_rng(seed)

    def select(self, action_ids: List[str], round_id: int) -> str:
        idx = self._rng.integers(0, len(action_ids))
        return action_ids[idx]

    def update(self, losses: List[float], round_id: int) -> None:
        pass  # No learning

    def get_weights(self) -> np.ndarray:
        return np.ones(self.num_agents) / self.num_agents

    def reset(self) -> None:
        pass  # No state

    @property
    def name(self) -> str:
        return "random_dictator"
