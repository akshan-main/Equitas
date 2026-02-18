"""YAML-backed dataclass configuration with validation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import AdversaryType, AggregatorMethod, CorruptionRealization, EnvironmentType


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    max_retries: int = 5
    initial_backoff: float = 5.0


@dataclass
class WorldConfig:
    crisis_axes: List[str] = field(default_factory=lambda: [
        "resource_scarcity", "external_threat", "inequality", "economic_instability",
    ])
    policy_dims: List[str] = field(default_factory=lambda: [
        "tax_merchants", "welfare_workers", "military_spend", "education_investment",
    ])
    actions_per_crisis: int = 3
    num_rounds: int = 40


@dataclass
class CommitteeConfig:
    class_ids: List[str] = field(default_factory=lambda: [
        "guardian", "auxiliary", "producer",
    ])
    members_per_class: int = 7
    num_judges: int = 5


@dataclass
class CorruptionConfig:
    corruption_rate: float = 0.25
    adversary_type: str = "selfish"
    corruption_realization: str = "algorithmic"  # "algorithmic" | "llm"
    corruption_onset_round: Optional[int] = None
    coordinated_target: str = "worst_city"
    scheduled_honest_rounds: int = 10  # for scheduled (delayed-onset) adversary
    deceptive_strength: str = "strong"
    corruption_target: str = "members"  # "members" | "judges" | "both"


@dataclass
class AggregatorConfig:
    method: str = "multiplicative_weights"
    eta: float = 1.0
    min_weight: float = 1e-6
    ema_alpha: float = 0.3
    trim_fraction: float = 0.2
    num_samples: int = 5
    alpha: float = 1.0
    beta: float = 0.5


@dataclass
class ExperimentConfig:
    name: str = "default"
    environment: str = "governance"
    seed: int = 42
    num_runs: int = 3

    llm: LLMConfig = field(default_factory=LLMConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    committee: CommitteeConfig = field(default_factory=CommitteeConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)

    aggregators: List[AggregatorConfig] = field(default_factory=lambda: [
        AggregatorConfig(method="majority_vote"),
        AggregatorConfig(method="oracle_upper_bound"),
        AggregatorConfig(method="self_consistency", num_samples=5),
        AggregatorConfig(method="ema_trust", ema_alpha=0.3),
        AggregatorConfig(method="trimmed_vote", trim_fraction=0.2),
        AggregatorConfig(method="multiplicative_weights", eta=1.0),
        AggregatorConfig(method="confidence_weighted"),
        AggregatorConfig(method="random_dictator"),
        AggregatorConfig(method="supervisor_rerank"),
    ])

    corruption_rates: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    adversary_types: List[str] = field(default_factory=lambda: [
        "selfish", "coordinated", "scheduled", "deceptive",
    ])
    alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    beta_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    committee_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 10])

    gsm8k_data_path: str = "data/gsm8k_test.csv"
    gsm8k_max_examples: int = 50

    output_dir: str = "results"
    save_conversation_log: bool = True

    experiment_type: str = "sweep"


def _dataclass_from_dict(cls: type, d: Dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict."""
    if not isinstance(d, dict):
        return d
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            # Handle nested dataclasses
            if isinstance(ft, str):
                ft = globals().get(ft) or locals().get(ft)
            if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__") and isinstance(v, dict):
                kwargs[k] = _dataclass_from_dict(ft, v)
            elif isinstance(v, list) and k == "aggregators":
                kwargs[k] = [_dataclass_from_dict(AggregatorConfig, item) if isinstance(item, dict) else item for item in v]
            else:
                kwargs[k] = v
    return cls(**kwargs)


# Mapping of field names to their dataclass types for nested construction
_NESTED_FIELDS = {
    "llm": LLMConfig,
    "world": WorldConfig,
    "committee": CommitteeConfig,
    "corruption": CorruptionConfig,
}


def load_config(path: str) -> ExperimentConfig:
    """Load ExperimentConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    kwargs = {}
    for k, v in raw.items():
        if k in _NESTED_FIELDS and isinstance(v, dict):
            kwargs[k] = _dataclass_from_dict(_NESTED_FIELDS[k], v)
        elif k == "aggregators" and isinstance(v, list):
            kwargs[k] = [
                _dataclass_from_dict(AggregatorConfig, item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            kwargs[k] = v

    return ExperimentConfig(**kwargs)


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save config to YAML for reproducibility."""
    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        if isinstance(obj, Enum):
            return obj.value
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False, sort_keys=False)


def validate_config(config: ExperimentConfig) -> None:
    """Validate config constraints."""
    assert 0.0 <= config.corruption.corruption_rate <= 1.0, "corruption_rate must be in [0, 1]"
    assert config.committee.members_per_class >= 1, "members_per_class must be >= 1"
    assert config.committee.num_judges >= 1, "num_judges must be >= 1"
    assert config.world.num_rounds >= 1, "num_rounds must be >= 1"
    assert config.world.actions_per_crisis >= 2, "actions_per_crisis must be >= 2"
    assert config.corruption.adversary_type in [e.value for e in AdversaryType], (
        f"Invalid adversary_type: {config.corruption.adversary_type}"
    )
    assert config.corruption.corruption_realization in [e.value for e in CorruptionRealization], (
        f"Invalid corruption_realization: {config.corruption.corruption_realization}. "
        f"Valid: {[e.value for e in CorruptionRealization]}"
    )
    assert config.environment in [e.value for e in EnvironmentType], (
        f"Invalid environment: {config.environment}"
    )
    for agg in config.aggregators:
        assert agg.method in [e.value for e in AggregatorMethod], (
            f"Invalid aggregator method: {agg.method}"
        )
