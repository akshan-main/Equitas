"""Oracle upper bound: best single agent in hindsight (not implementable online)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class OracleUpperBoundAggregator(BaseAggregator):
    """
    NOT a baseline — this is an upper bound.
    During simulation: uses majority vote as placeholder.
    Post-hoc: retrospectively uses the single agent with lowest cumulative loss.
    """

    mechanism_meta = MechanismMeta(
        assumption="Full hindsight — knows which agent was best after all rounds",
        works_when=["Always (by definition, as an upper bound)"],
        breaks_when=["Not implementable online — reference only"],
        is_online=False,
        is_oracle=True,
    )

    def __init__(self, num_agents: int) -> None:
        super().__init__(num_agents)
        self._cumulative_losses = np.zeros(num_agents)
        self._history: List[List[str]] = []

    def select(self, action_ids: List[str], round_id: int) -> str:
        self._history.append(list(action_ids))
        w = np.ones(self.num_agents) / self.num_agents
        return weighted_vote(action_ids, w)

    def update(self, losses: List[float], round_id: int) -> None:
        self._cumulative_losses += np.array(losses)

    def retrospective_decisions(self) -> List[str]:
        """Post-hoc: what the best single agent chose each round."""
        best_idx = int(np.argmin(self._cumulative_losses))
        return [recs[best_idx] for recs in self._history]

    def get_weights(self) -> np.ndarray:
        w = np.zeros(self.num_agents)
        if self._cumulative_losses.sum() > 0:
            w[np.argmin(self._cumulative_losses)] = 1.0
        else:
            w[:] = 1.0 / self.num_agents
        return w

    def reset(self) -> None:
        self._cumulative_losses = np.zeros(self.num_agents)
        self._history = []

    @property
    def name(self) -> str:
        return "oracle_upper_bound"
