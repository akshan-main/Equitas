"""All shared dataclasses, enums, and type aliases."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ClassID(str, Enum):
    GUARDIAN = "guardian"
    AUXILIARY = "auxiliary"
    PRODUCER = "producer"


class AdversaryType(str, Enum):
    NONE = "none"
    SELFISH = "selfish"
    COORDINATED = "coordinated"
    SCHEDULED = "scheduled"  # delayed-onset contamination (Huber-compliant)
    DECEPTIVE = "deceptive"


class CorruptionRealization(str, Enum):
    """How the corrupted distribution Q is sampled."""
    ALGORITHMIC = "algorithmic"  # worst-case, deterministic from world model
    LLM = "llm"  # realistic, LLM with adversarial prompt


class AggregatorMethod(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    ORACLE_UPPER_BOUND = "oracle_upper_bound"
    SELF_CONSISTENCY = "self_consistency"
    EMA_TRUST = "ema_trust"
    TRIMMED_VOTE = "trimmed_vote"
    MULTIPLICATIVE_WEIGHTS = "multiplicative_weights"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RANDOM_DICTATOR = "random_dictator"
    SUPERVISOR_RERANK = "supervisor_rerank"


class EnvironmentType(str, Enum):
    GOVERNANCE = "governance"
    GSM8K = "gsm8k"


@dataclass
class ActionSpec:
    """A policy action the city can take in a given crisis."""
    id: str
    description: str
    policy: Dict[str, float]


@dataclass
class Crisis:
    """A crisis scenario facing the city."""
    id: int
    axes: Dict[str, float]
    description: str
    actions: List[ActionSpec]


@dataclass
class Outcome:
    """Full outcome metrics for a (crisis, action) pair."""
    crisis_id: int
    action_id: str
    class_utilities: Dict[str, float]
    city_utility: float
    unfairness: float
    fairness_jain: float
    worst_group_utility: float


@dataclass
class GSM8KItem:
    """Single GSM8K example."""
    id: int
    question: str
    full_answer: str
    short_answer: str
    group: str  # difficulty bucket: short/medium/long


@dataclass
class AgentRecommendation:
    """What a single agent recommended for a single round."""
    agent_id: str
    class_id: str
    action_id: str
    rationale: str
    corrupted: bool
    adversary_type: AdversaryType


@dataclass
class RoundResult:
    """All data for one round of the simulation."""
    round_id: int
    crisis: Any  # Crisis or GSM8KItem
    oracle_action_id: str
    oracle_outcome: Outcome

    class_member_recs: Dict[str, List[AgentRecommendation]]
    class_leader_proposals: Dict[str, AgentRecommendation]

    judge_evaluations: List[AgentRecommendation]

    aggregator_decisions: Dict[str, str]
    aggregator_outcomes: Dict[str, Outcome]

    mw_weights_snapshot: Optional[Dict[str, np.ndarray]] = None

    # Full-hierarchy mode: per-aggregator caches for recording/replay
    # leader_call_cache: "cls_id:action_id" -> AgentRecommendation
    leader_call_cache: Optional[Dict[str, AgentRecommendation]] = None
    # judge_call_cache: proposal_set_key -> List[AgentRecommendation]
    judge_call_cache: Optional[Dict[str, List[AgentRecommendation]]] = None
    # agg_proposal_keys: aggregator_name -> proposal_set_key
    agg_proposal_keys: Optional[Dict[str, str]] = None


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""
    config: Any
    rounds: List[RoundResult]
    aggregator_log: pd.DataFrame
    agent_log: pd.DataFrame
    weight_history: Dict[str, List[np.ndarray]]
