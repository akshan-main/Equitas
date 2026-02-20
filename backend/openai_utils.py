# backend/openai_utils.py
from __future__ import annotations

import asyncio
from typing import Any
import openai  # this is the official `openai` package

async def run_with_backoff(agent, task: str, max_retries: int = 5):
    """
    Call agent.run(task=...) with simple exponential backoff on rate limits.
    """
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return await agent.run(task=task)
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Hit rate limit (attempt {attempt+1}), backing off...")
            await asyncio.sleep(delay)
            delay *= 2.0
