"""Abstract aggregator interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class MechanismMeta:
    """Mechanism assumptions and regime metadata for paper taxonomy."""

    assumption: str
    works_when: List[str] = field(default_factory=list)
    breaks_when: List[str] = field(default_factory=list)
    is_online: bool = False
    is_oracle: bool = False


class BaseAggregator(ABC):
    """Common interface for all aggregation methods."""

    mechanism_meta: MechanismMeta  # subclasses must set this

    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents

    @abstractmethod
    def select(self, action_ids: List[str], round_id: int) -> str:
        """Given agent recommendations, select the final action."""
        ...

    @abstractmethod
    def update(self, losses: List[float], round_id: int) -> None:
        """Update internal state after seeing per-agent losses."""
        ...

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Return current weight vector."""
        ...

    def reset(self) -> None:
        """Reset state for a new simulation run."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...


def weighted_vote(action_ids: List[str], weights: np.ndarray) -> str:
    """Weighted plurality vote. Deterministic tie-break: alphabetical."""
    tallies: dict[str, float] = {}
    for aid, w in zip(action_ids, weights):
        tallies[aid] = tallies.get(aid, 0.0) + w
    max_weight = max(tallies.values())
    winners = [aid for aid, w in tallies.items() if abs(w - max_weight) < 1e-12]
    return sorted(winners)[0]
