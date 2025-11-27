"""
Robust statistics and fairness utilities used for evaluation and analysis.

- trimmed mean
- median absolute deviation (MAD)
- simple bootstrap CIs
- Jain's fairness index
- Gini coefficient
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


def trimmed_mean(values, alpha: float = 0.1) -> float:
    """
    Compute alpha-trimmed mean: drop alpha fraction from each tail.
    alpha in [0, 0.5).
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.shape[0]
    if n == 0:
        raise ValueError("trimmed_mean: empty array")
    k = int(alpha * n)
    if 2 * k >= n:
        # If too few points to trim, revert to plain mean
        return float(arr.mean())
    trimmed = arr[k : n - k]
    return float(trimmed.mean())


def mad(values) -> float:
    """
    Median absolute deviation (unscaled).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("mad: empty array")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def bootstrap_ci(
    values,
    estimator: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Simple percentile bootstrap CI for an estimator.

    Returns (lower, upper).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: empty array")

    if rng is None:
        rng = np.random.default_rng()

    n = arr.shape[0]
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        estimates[i] = estimator(sample)

    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lower, upper


def fairness_index_jain(values) -> float:
    """
    Jain's fairness index in [0, 1], higher is fairer.

    J(x) = (sum_i x_i)^2 / (n * sum_i x_i^2)
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("fairness_index_jain: empty array")
    s1 = float(arr.sum())
    s2 = float((arr ** 2).sum())
    if s2 == 0.0:
        # everyone at zero -> undefined; treat as perfectly fair but bad welfare
        return 1.0
    n = arr.size
    return (s1 * s1) / (n * s2)


def gini_coefficient(values) -> float:
    """
    Gini coefficient in [0, 1]. 0 = perfect equality, 1 = maximal inequality.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("gini_coefficient: empty array")
    if np.all(arr == 0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(sorted_arr, dtype=float)
    gini = 1.0 - 2.0 * (cum.sum() / (cum[-1] * n) - (n + 1) / (2.0 * n))
    return float(gini)
