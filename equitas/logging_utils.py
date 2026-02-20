"""JSONL conversation logger for audit/reproducibility."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


_LOG_PATH: str | None = None


def _get_log_path() -> str:
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = os.environ.get("EQUITAS_CONV_LOG", "results/conversation_log.jsonl")
    return _LOG_PATH


def set_log_path(path: str) -> None:
    global _LOG_PATH
    _LOG_PATH = path


def log_conversation(entry: Dict[str, Any]) -> None:
    """Append a JSON entry to the conversation log."""
    path = _get_log_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
