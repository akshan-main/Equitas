"""Corruption onset scheduling for mid-run corruption experiments."""
from __future__ import annotations

from typing import List, Set


class CorruptionSchedule:
    """
    Manages when corruption activates during a simulation.
    - onset_round=0: corrupt from start
    - onset_round=K: honest for rounds 0..K-1, corrupt from K onward
    """

    def __init__(
        self,
        onset_round: int = 0,
        corrupted_agent_ids: List[str] | None = None,
    ) -> None:
        self.onset_round = onset_round
        self.corrupted_agent_ids: Set[str] = set(corrupted_agent_ids or [])

    def is_active(self, round_id: int) -> bool:
        return round_id >= self.onset_round

    def should_be_corrupted(self, agent_id: str, round_id: int) -> bool:
        """Check if a specific agent should be corrupted at this round."""
        return agent_id in self.corrupted_agent_ids and self.is_active(round_id)

    def apply_to_agents(self, agents: list, round_id: int) -> None:
        """
        Toggle agent.corrupted based on schedule and round.
        Agents whose IDs are in corrupted_agent_ids get corrupted
        only when the schedule is active.
        """
        active = self.is_active(round_id)
        for agent in agents:
            if agent.agent_id in self.corrupted_agent_ids:
                agent.corrupted = active
            # Agents not in the corrupted set remain as-is
