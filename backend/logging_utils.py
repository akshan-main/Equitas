# backend/logging_utils.py

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

# You can change the path via env var if you want:
#   EQUITAS_CONV_LOG=results/kalilopolis_conversations.jsonl python -m backend.experiments
_LOG_PATH = os.getenv(
    "EQUITAS_CONV_LOG",
    os.path.join("results", "conversation_log.jsonl"),
)


def log_conversation(entry: Dict[str, Any]) -> None:
    """
    Append a single conversation record as a JSON line.

    We automatically add a UTC timestamp if missing.
    """
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

    record = dict(entry)
    record.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
