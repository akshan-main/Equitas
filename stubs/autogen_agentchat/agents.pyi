from __future__ import annotations

from typing import Any, Awaitable, Protocol


class BaseChatMessage(Protocol):
    content: Any


class TaskResult(Protocol):
    messages: list[BaseChatMessage]


class AssistantAgent:
    def __init__(self, *, name: str, model_client: Any, system_message: str) -> None: ...

    async def run(self, *, task: str) -> TaskResult: ...

