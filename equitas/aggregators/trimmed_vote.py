"""Method 5: Trimmed vote (drop outlier agents by cumulative loss, then vote)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class TrimmedVoteAggregator(BaseAggregator):
    """
    Tracks cumulative loss per agent. Each round, drops the top
    trim_fraction of agents by cumulative loss, then majority votes
    among the rest.
    """

    mechanism_meta = MechanismMeta(
        assumption="Corrupted agents have identifiably higher cumulative loss",
        works_when=[
            "Corruption produces consistently high-loss recommendations",
            "trim_fraction ≥ ε (enough capacity to exclude all adversaries)",
            "High-outlier regime where adversaries stand out from honest agents",
        ],
        breaks_when=[
            "Delayed-onset corruption (adversary builds low cumulative loss first)",
            "trim_fraction < ε (not enough trimming to cover all adversaries)",
            "Adversary loss is similar to honest loss (selfish in aligned settings)",
        ],
        is_online=True,
    )

    def __init__(self, num_agents: int, trim_fraction: float = 0.2) -> None:
        super().__init__(num_agents)
        self.trim_fraction = trim_fraction
        self._cumulative_loss = np.zeros(num_agents)

    def select(self, action_ids: List[str], round_id: int) -> str:
        n_trim = int(self.num_agents * self.trim_fraction)
        if n_trim > 0 and self._cumulative_loss.sum() > 0:
            # Trim the agents with highest cumulative loss
            sorted_idx = np.argsort(self._cumulative_loss)
            keep = sorted_idx[: self.num_agents - n_trim]
        else:
            keep = np.arange(self.num_agents)

        weights = np.zeros(self.num_agents)
        weights[keep] = 1.0 / len(keep)
        return weighted_vote(action_ids, weights)

    def update(self, losses: List[float], round_id: int) -> None:
        self._cumulative_loss += np.array(losses)

    def get_weights(self) -> np.ndarray:
        n_trim = int(self.num_agents * self.trim_fraction)
        if n_trim > 0 and self._cumulative_loss.sum() > 0:
            sorted_idx = np.argsort(self._cumulative_loss)
            keep = sorted_idx[: self.num_agents - n_trim]
            weights = np.zeros(self.num_agents)
            weights[keep] = 1.0 / len(keep)
            return weights
        return np.ones(self.num_agents) / self.num_agents

    def reset(self) -> None:
        self._cumulative_loss = np.zeros(self.num_agents)

    @property
    def name(self) -> str:
        return "trimmed_vote"
