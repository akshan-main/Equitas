"""Abstract adversary interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..types import Crisis


class BaseAdversary(ABC):
    """Determines what a corrupted agent does."""

    @abstractmethod
    def corrupt_recommendation(
        self,
        crisis: Crisis,
        class_id: str,
        honest_action_id: str,
        round_id: int,
        context: Optional[Dict] = None,
    ) -> str:
        """Return the corrupted action_id."""
        ...

    def corrupt_rationale(
        self,
        crisis: Crisis,
        action_id: str,
        class_id: str,
    ) -> Optional[str]:
        """Return a corrupted rationale, or None to use default."""
        return None

    def reset(self) -> None:
        """Reset adversary state between runs."""
        pass
