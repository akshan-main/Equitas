"""Method 9: Supervisor-rerank (follow-the-leader, delegates to best agent).

Models the "supervisor/manager" pattern common in production multi-agent
systems: a central coordinator tracks agent quality and delegates to
the currently best-performing agent each round.

Algorithmically, this is Follow-the-Leader (FTL): each round, pick the
agent with lowest cumulative loss and use their recommendation.
No weighting, no voting — all-in on one agent.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseAggregator, MechanismMeta


class SupervisorRerankAggregator(BaseAggregator):
    """Follow-the-leader: use the recommendation of the best agent so far.

    Models the real-world pattern where a supervisor agent tracks
    sub-agent performance and delegates to the current best performer.
    """

    mechanism_meta = MechanismMeta(
        assumption="Past cumulative loss identifies the most reliable agent",
        works_when=[
            "One agent is consistently best — supervisor locks onto it quickly",
            "Stable corruption: honest agents stay honest, corrupt stay corrupt",
            "Low noise: cumulative loss clearly separates honest from corrupt",
        ],
        breaks_when=[
            "Scheduled adversaries build trust then exploit — once leader, full control",
            "Multiple agents have similar loss (noisy leader selection)",
            "All-in commitment means one bad round from the leader = full damage",
        ],
        is_online=True,
    )

    def __init__(self, num_agents: int) -> None:
        super().__init__(num_agents)
        self._cumulative_loss = np.zeros(num_agents)

    def select(self, action_ids: List[str], round_id: int) -> str:
        if self._cumulative_loss.sum() == 0:
            # No history yet — use first agent (deterministic)
            return action_ids[0]
        leader_idx = int(np.argmin(self._cumulative_loss))
        return action_ids[leader_idx]

    def update(self, losses: List[float], round_id: int) -> None:
        self._cumulative_loss += np.array(losses)

    def get_weights(self) -> np.ndarray:
        w = np.zeros(self.num_agents)
        if self._cumulative_loss.sum() > 0:
            w[np.argmin(self._cumulative_loss)] = 1.0
        else:
            w[:] = 1.0 / self.num_agents
        return w

    def reset(self) -> None:
        self._cumulative_loss = np.zeros(self.num_agents)

    @property
    def name(self) -> str:
        return "supervisor_rerank"
