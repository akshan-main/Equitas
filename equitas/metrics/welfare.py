"""Welfare and performance metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .fairness import jain_fairness_index


def city_utility(class_utilities: Dict[str, float]) -> float:
    """Mean of class utilities."""
    return float(np.mean(list(class_utilities.values())))


def regret(city_util: float, oracle_city_util: float) -> float:
    """Regret = oracle - actual. Non-negative."""
    return max(0.0, oracle_city_util - city_util)


def combined_loss(
    city_util: float,
    oracle_city_util: float,
    fairness_jain: float,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Combined loss = alpha * regret + beta * (1 - fairness)."""
    r = regret(city_util, oracle_city_util)
    f_penalty = 1.0 - fairness_jain
    return alpha * r + beta * f_penalty


def accuracy(predicted: str, gold: str) -> float:
    """Binary accuracy for GSM8K (exact string match after strip)."""
    return 1.0 if predicted.strip() == gold.strip() else 0.0
