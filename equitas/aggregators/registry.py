"""Aggregator registry: name -> class mapping, factory function."""
from __future__ import annotations

from typing import Dict, Type

from ..config import AggregatorConfig
from .base import BaseAggregator
from .best_single import OracleUpperBoundAggregator
from .confidence_weighted import ConfidenceWeightedVoteAggregator
from .ema_trust import EMATrustAggregator
from .majority_vote import MajorityVoteAggregator
from .multiplicative_weights import MultiplicativeWeightsAggregator
from .random_dictator import RandomDictatorAggregator
from .self_consistency import SelfConsistencyAggregator
from .supervisor_rerank import SupervisorRerankAggregator
from .trimmed_vote import TrimmedVoteAggregator

AGGREGATOR_REGISTRY: Dict[str, Type[BaseAggregator]] = {
    "majority_vote": MajorityVoteAggregator,
    "oracle_upper_bound": OracleUpperBoundAggregator,
    "self_consistency": SelfConsistencyAggregator,
    "ema_trust": EMATrustAggregator,
    "trimmed_vote": TrimmedVoteAggregator,
    "multiplicative_weights": MultiplicativeWeightsAggregator,
    "confidence_weighted": ConfidenceWeightedVoteAggregator,
    "random_dictator": RandomDictatorAggregator,
    "supervisor_rerank": SupervisorRerankAggregator,
}


def create_aggregator(config: AggregatorConfig, num_agents: int) -> BaseAggregator:
    """Instantiate an aggregator from config."""
    cls = AGGREGATOR_REGISTRY[config.method]
    if config.method == "multiplicative_weights":
        return cls(
            num_agents,
            eta=config.eta,
            alpha=config.alpha,
            beta=config.beta,
            min_weight=config.min_weight,
        )
    elif config.method == "ema_trust":
        return cls(num_agents, ema_alpha=config.ema_alpha)
    elif config.method == "trimmed_vote":
        return cls(num_agents, trim_fraction=config.trim_fraction)
    elif config.method == "self_consistency":
        return cls(config.num_samples)
    else:
        return cls(num_agents)
