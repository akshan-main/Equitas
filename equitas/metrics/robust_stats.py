"""Robust statistics: trimmed mean, MAD, bootstrap CI."""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np


def trimmed_mean(values: Union[list, np.ndarray], alpha: float = 0.1) -> float:
    """Trimmed mean: drop alpha fraction from each tail."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0:
        return 0.0
    lo = int(n * alpha)
    hi = n - lo
    if hi <= lo:
        return float(np.mean(arr))
    return float(np.mean(arr[lo:hi]))


def mad(values: Union[list, np.ndarray]) -> float:
    """Median absolute deviation (unscaled)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def bootstrap_ci(
    values: Union[list, np.ndarray],
    estimator: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    if rng is None:
        rng = np.random.default_rng(0)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        estimates[i] = estimator(sample)
    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return (lo, hi)
