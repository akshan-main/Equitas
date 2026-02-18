"""Governor: pure algorithmic aggregator over judge opinions. No LLM."""
from __future__ import annotations

from typing import List

import numpy as np

from ..aggregators.base import BaseAggregator
from ..types import AgentRecommendation


class Governor:
    """
    The governor is NOT an LLM agent. It is a pure algorithmic
    aggregator that combines judge recommendations.
    """

    def __init__(self, aggregator: BaseAggregator) -> None:
        self.aggregator = aggregator

    def decide(
        self,
        judge_recs: List[AgentRecommendation],
        round_id: int,
    ) -> str:
        action_ids = [rec.action_id for rec in judge_recs]
        return self.aggregator.select(action_ids, round_id)

    def update(self, losses: List[float], round_id: int) -> None:
        self.aggregator.update(losses, round_id)

    def get_weights(self) -> np.ndarray:
        return self.aggregator.get_weights()

    def reset(self) -> None:
        self.aggregator.reset()
