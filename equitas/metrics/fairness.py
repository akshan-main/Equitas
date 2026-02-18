"""Fairness metrics for multi-class outcomes."""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np


def jain_fairness_index(values: Union[List[float], np.ndarray]) -> float:
    """Jain's fairness index in [1/n, 1]. Higher = fairer."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    return float((arr.sum() ** 2) / (len(arr) * (arr ** 2).sum()))


def worst_group_utility(class_utilities: Dict[str, float]) -> float:
    """Return minimum class utility (Rawlsian fairness)."""
    return min(class_utilities.values())


def unfairness_gap(class_utilities: Dict[str, float]) -> float:
    """Return max - min class utility."""
    vals = list(class_utilities.values())
    return max(vals) - min(vals)


def gini_coefficient(values: Union[List[float], np.ndarray]) -> float:
    """Gini coefficient in [0, 1]. 0 = perfect equality."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum() / (n * arr.sum())) - (n + 1) / n)
