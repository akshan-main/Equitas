"""Seeding, path helpers, misc utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a reproducible numpy random generator."""
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *components: object) -> int:
    """Derive a deterministic seed from a base seed and extra components.

    Uses stable hashing so seeds are reproducible across Python processes.
    """
    payload = json.dumps([base_seed, *components], sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
