from __future__ import annotations

from typing import Protocol


class OpenAIChatCompletionClient:
    def __init__(self, *, model: str) -> None: ...

    def __call__(self, *args: object, **kwargs: object) -> object: ...

