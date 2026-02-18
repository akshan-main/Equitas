"""Pareto frontier computation for welfare-fairness tradeoff."""
from __future__ import annotations

from typing import List

import numpy as np


def pareto_frontier_indices(
    welfare: np.ndarray,
    fairness: np.ndarray,
) -> np.ndarray:
    """
    Return indices of Pareto-optimal points (higher is better on both axes).
    """
    n = len(welfare)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if welfare[j] >= welfare[i] and fairness[j] >= fairness[i]:
                if welfare[j] > welfare[i] or fairness[j] > fairness[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def pareto_sweep_results(
    alpha_values: List[float],
    beta_values: List[float],
    welfare_matrix: np.ndarray,
    fairness_matrix: np.ndarray,
) -> List[dict]:
    """
    Given matrices indexed by (alpha_idx, beta_idx) containing mean welfare
    and fairness, return a list of dicts for plotting.
    """
    results = []
    for i, a in enumerate(alpha_values):
        for j, b in enumerate(beta_values):
            results.append({
                "alpha": a,
                "beta": b,
                "welfare": float(welfare_matrix[i, j]),
                "fairness": float(fairness_matrix[i, j]),
            })
    return results
