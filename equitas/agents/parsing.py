"""Utilities for parsing structured responses from LLMs."""
from __future__ import annotations

import re
from typing import List, Optional


def parse_action_id(text: str, valid_ids: List[str]) -> str:
    """
    Parse ACTION_ID: <X> from LLM text.
    Fallback chain: regex match -> any valid id in text -> first valid id.
    """
    # Try explicit ACTION_ID pattern
    match = re.search(r"ACTION_ID\s*:\s*(\S+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().strip(".")
        if candidate in valid_ids:
            return candidate

    # Try "Action X" pattern
    match = re.search(r"Action\s+(\S+)", text)
    if match:
        candidate = match.group(1).strip().strip(".:)")
        if candidate in valid_ids:
            return candidate

    # Search for any valid ID in the text (last occurrence, as LLMs often conclude)
    for vid in reversed(valid_ids):
        pattern = r"\b" + re.escape(vid) + r"\b"
        if re.search(pattern, text):
            return vid

    # Fallback: first valid ID
    return valid_ids[0] if valid_ids else "A"


def extract_rationale(text: str) -> str:
    """Extract reasoning text before the ACTION_ID line."""
    lines = text.strip().split("\n")
    rationale_lines = []
    for line in lines:
        if re.match(r"ACTION_ID\s*:", line, re.IGNORECASE):
            break
        rationale_lines.append(line)
    return "\n".join(rationale_lines).strip()


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from math solution text."""
    # Try #### pattern first (GSM8K format)
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "answer is X" pattern
    match = re.search(r"(?:answer|result)\s+(?:is|=)\s*([\d,.\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try last number in text
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None
