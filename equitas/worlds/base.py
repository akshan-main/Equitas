"""Abstract base class for environments/worlds."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np

from ..types import Outcome


class BaseWorld(ABC):

    @abstractmethod
    def generate_rounds(self, num_rounds: int, rng: np.random.Generator) -> List[Any]:
        ...

    @abstractmethod
    def evaluate_action(self, round_data: Any, action_id: str) -> Outcome:
        ...

    @abstractmethod
    def best_action(self, round_data: Any) -> Tuple[str, Outcome]:
        ...

    @abstractmethod
    def get_action_ids(self, round_data: Any) -> List[str]:
        ...

    @abstractmethod
    def format_round_for_prompt(self, round_data: Any) -> str:
        ...
