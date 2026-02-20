"""LLM client wrapper with retry logic. Supports any OpenAI-compatible API."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

import openai

from ..config import LLMConfig


class LLMClientWrapper:
    """Async OpenAI-compatible chat completions with exponential backoff.

    Uses sync OpenAI client + asyncio.to_thread for compatibility with
    development httpx versions that don't support event_hooks=None.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env, "")
        kwargs: dict = {"api_key": api_key}
        if config.api_base:
            kwargs["base_url"] = config.api_base
        self.client = openai.OpenAI(**kwargs)

    def _sync_complete(
        self,
        system_message: str,
        user_message: str,
        temp: float,
    ) -> str:
        """Blocking call — run via asyncio.to_thread."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temp,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def complete(
        self,
        system_message: str,
        user_message: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a chat completion and return the response text."""
        temp = temperature if temperature is not None else self.config.temperature
        delay = self.config.initial_backoff

        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.to_thread(
                    self._sync_complete, system_message, user_message, temp,
                )
            except (openai.RateLimitError, openai.APIConnectionError):
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2.0
            except openai.APIError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2.0
