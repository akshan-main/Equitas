"""Method 7: Confidence-weighted vote (harmonic decay from cumulative loss)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta, weighted_vote


class ConfidenceWeightedVoteAggregator(BaseAggregator):
    """w_i = 1 / (1 + cumulative_loss_i), then normalize.

    Harmonic decay: bad rounds accumulate but never fully destroy weight.
    Different from EMA (recency-biased) and MW (multiplicative penalty).

    In a full deployment, confidence would come from agent self-reports.
    Self-reported confidence creates an adversarial surface:
    adversaries can inflate confidence to gain disproportionate weight.
    """

    mechanism_meta = MechanismMeta(
        assumption="Cumulative loss is a reliable proxy for agent quality",
        works_when=[
            "Corruption produces consistently higher losses than honest agents",
            "Agents with low cumulative loss are more trustworthy",
            "Harmonic decay is more forgiving than MW — slower to exclude agents",
        ],
        breaks_when=[
            "Adversaries inflate self-reported confidence (if using self-reports)",
            "Delayed-onset corruption: honest phase builds low cumulative loss",
            "All agents have similar loss profiles (no separation signal)",
        ],
        is_online=True,
    )

    def __init__(self, num_agents: int) -> None:
        super().__init__(num_agents)
        self._cumulative_loss = np.zeros(num_agents)

    def select(self, action_ids: List[str], round_id: int) -> str:
        return weighted_vote(action_ids, self._weights_from_loss())

    def update(self, losses: List[float], round_id: int) -> None:
        self._cumulative_loss += np.array(losses)

    def _weights_from_loss(self) -> np.ndarray:
        raw = 1.0 / (1.0 + self._cumulative_loss)
        total = raw.sum()
        if total > 0:
            return raw / total
        return np.ones(self.num_agents) / self.num_agents

    def get_weights(self) -> np.ndarray:
        return self._weights_from_loss()

    def reset(self) -> None:
        self._cumulative_loss = np.zeros(self.num_agents)

    @property
    def name(self) -> str:
        return "confidence_weighted"
