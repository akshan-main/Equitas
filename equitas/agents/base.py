"""Abstract base class for all agent types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import AdversaryType, AgentRecommendation


class BaseAgent(ABC):

    def __init__(
        self,
        agent_id: str,
        class_id: str,
        corrupted: bool = False,
        adversary_type: AdversaryType = AdversaryType.NONE,
    ) -> None:
        self.agent_id = agent_id
        self.class_id = class_id
        self.corrupted = corrupted
        self.adversary_type = adversary_type

    @abstractmethod
    async def act(self, *args: Any, **kwargs: Any) -> AgentRecommendation:
        ...
