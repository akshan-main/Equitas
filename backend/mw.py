"""
Multiplicative Weights aggregator for advisors.

- Maintains a weight per advisor.
- Uses weighted vote over action IDs.
- Updates weights based on per-advisor losses (city regret + fairness penalty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


def weighted_vote(
    recommended_actions: List[str],
    weights: np.ndarray,
) -> str:
    """
    Weighted majority vote over discrete action IDs.
    """
    if len(recommended_actions) == 0:
        raise ValueError("No recommendations provided.")

    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != len(recommended_actions):
        raise ValueError("weights and recommended_actions length mismatch.")

    vote_scores = {}
    for action, w in zip(recommended_actions, weights):
        vote_scores[action] = vote_scores.get(action, 0.0) + float(w)

    # Argmax over actions by score
    best_action = max(vote_scores.items(), key=lambda kv: kv[1])[0]
    return best_action


@dataclass
class MWConfig:
    """
    Hyperparameters for multiplicative weights.

    eta: learning rate (called β in the paper text).
    """
    eta: float = 1.0       # learning rate
    alpha: float = 1.0     # weight on city_regret (used in loss construction)
    beta: float = 0.5      # weight on fairness_penalty (used in loss construction)
    min_weight: float = 1e-6


@dataclass
class MWState:
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


class MultiplicativeWeightsAggregator:
    """
    Online multiplicative weights aggregator.

    At each round:
    - receives advisors' recommended actions
    - chooses an action via weighted vote
    - after seeing losses for each advisor, updates weights
    """

    def __init__(self, num_advisors: int, config: MWConfig):
        self.config = config
        self.state = MWState(weights=np.ones(num_advisors, dtype=float) / num_advisors)

    def select_action(self, recommended_actions: List[str]) -> str:
        return weighted_vote(recommended_actions, self.state.weights)

    def update(self, losses: List[float]) -> None:
        """
        losses[i] is loss for advisor i. Lower is better.
        w_i <- w_i * exp(-eta * loss_i), then renormalize.
        """
        losses_arr = np.asarray(losses, dtype=float)
        if losses_arr.shape[0] != self.state.weights.shape[0]:
            raise ValueError("Loss length != number of advisors.")

        eta = self.config.eta
        new_weights = self.state.weights * np.exp(-eta * losses_arr)

        # Avoid all-zero
        new_weights = np.maximum(new_weights, self.config.min_weight)

        new_weights /= new_weights.sum()
        self.state.weights = new_weights
