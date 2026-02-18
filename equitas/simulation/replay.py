"""
Record/replay system for LLM calls.

Record mode: run with real LLMs, save all agent recommendations to JSONL.
Replay mode: load saved recommendations, re-run aggregation only (zero API calls).

This lets you:
- Run LLMs once ($25), then iterate on aggregators for free
- Add new baselines without re-running any LLM calls
- Guarantee exact reproducibility
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..aggregators.base import BaseAggregator
from ..aggregators.best_single import OracleUpperBoundAggregator
from ..aggregators.registry import create_aggregator
from ..config import AggregatorConfig, ExperimentConfig
from ..metrics.fairness import jain_fairness_index
from ..metrics.welfare import combined_loss
from ..types import AdversaryType, AgentRecommendation, Outcome, SimulationResult
from ..worlds.base import BaseWorld


def save_recording(
    rounds_data: List[Dict[str, Any]],
    path: str,
) -> None:
    """Save all agent recommendations per round to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rd in rounds_data:
            f.write(json.dumps(rd) + "\n")


def load_recording(path: str) -> List[Dict[str, Any]]:
    """Load recorded agent recommendations from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_round(
    round_id: int,
    crisis_id: int,
    member_recs: Dict[str, List[Dict[str, Any]]],
    leader_proposals: Dict[str, Dict[str, Any]],
    judge_recs: List[Dict[str, Any]],
    oracle_action_id: str,
) -> Dict[str, Any]:
    """Create a recording entry for one round."""
    return {
        "round_id": round_id,
        "crisis_id": crisis_id,
        "oracle_action_id": oracle_action_id,
        "member_recs": member_recs,
        "leader_proposals": leader_proposals,
        "judge_recs": judge_recs,
    }


def rec_from_agent_rec(rec: AgentRecommendation) -> Dict[str, Any]:
    """Convert AgentRecommendation to serializable dict."""
    return {
        "agent_id": rec.agent_id,
        "class_id": rec.class_id,
        "action_id": rec.action_id,
        "rationale": rec.rationale,
        "corrupted": rec.corrupted,
        "adversary_type": rec.adversary_type.value,
    }


def agent_rec_from_dict(d: Dict[str, Any]) -> AgentRecommendation:
    """Reconstruct AgentRecommendation from dict."""
    return AgentRecommendation(
        agent_id=d["agent_id"],
        class_id=d["class_id"],
        action_id=d["action_id"],
        rationale=d.get("rationale", ""),
        corrupted=d.get("corrupted", False),
        adversary_type=AdversaryType(d.get("adversary_type", "none")),
    )


async def replay_simulation(
    config: ExperimentConfig,
    world: BaseWorld,
    recording_path: str,
) -> SimulationResult:
    """
    Re-run aggregation from recorded agent recommendations.
    Zero API calls. Lets you add/change aggregators for free.
    """
    records = load_recording(recording_path)
    crises = world.generate_rounds(len(records), np.random.default_rng(config.seed))

    # Create aggregators for judges
    num_judges = 0
    if records:
        num_judges = len(records[0].get("judge_recs", []))

    aggregators: Dict[str, BaseAggregator] = {}
    for agg_cfg in config.aggregators:
        aggregators[agg_cfg.method] = create_aggregator(agg_cfg, num_agents=num_judges)

    # Intra-class aggregators for members
    from ..aggregators.trimmed_vote import TrimmedVoteAggregator
    class_ids = config.committee.class_ids
    intra_aggs: Dict[str, BaseAggregator] = {}
    for cls_id in class_ids:
        if records and cls_id in records[0].get("member_recs", {}):
            n_members = len(records[0]["member_recs"][cls_id])
        else:
            n_members = config.committee.members_per_class
        intra_aggs[cls_id] = TrimmedVoteAggregator(n_members, trim_fraction=0.2)

    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    weight_history: Dict[str, List[np.ndarray]] = {name: [] for name in aggregators}

    for i, rec in enumerate(records):
        crisis = crises[i] if i < len(crises) else crises[-1]
        oracle_action_id = rec["oracle_action_id"]
        oracle_outcome = world.evaluate_action(crisis, oracle_action_id)

        # Reconstruct judge recommendations
        judge_recs = [agent_rec_from_dict(j) for j in rec.get("judge_recs", [])]
        judge_action_ids = [j.action_id for j in judge_recs]

        # Run each aggregator on the recorded judge votes
        for agg_name, agg in aggregators.items():
            chosen = agg.select(judge_action_ids, i)
            outcome = world.evaluate_action(crisis, chosen)

            agg_rows.append({
                "round_id": i,
                "aggregator": agg_name,
                "chosen_action_id": chosen,
                "city_utility": outcome.city_utility,
                "unfairness_gap": outcome.unfairness,
                "fairness_jain": outcome.fairness_jain,
                "worst_group_utility": outcome.worst_group_utility,
                "oracle_city_utility": oracle_outcome.city_utility,
                "regret": oracle_outcome.city_utility - outcome.city_utility,
            })

            # Compute judge losses and update
            agg_cfg = next(
                (c for c in config.aggregators if c.method == agg_name),
                AggregatorConfig(),
            )
            judge_losses = []
            for jrec in judge_recs:
                j_out = world.evaluate_action(crisis, jrec.action_id)
                loss = combined_loss(
                    j_out.city_utility, oracle_outcome.city_utility,
                    j_out.fairness_jain, alpha=agg_cfg.alpha, beta=agg_cfg.beta,
                )
                judge_losses.append(loss)
            agg.update(judge_losses, i)

        # Oracle row
        agg_rows.append({
            "round_id": i,
            "aggregator": "oracle",
            "chosen_action_id": oracle_action_id,
            "city_utility": oracle_outcome.city_utility,
            "unfairness_gap": oracle_outcome.unfairness,
            "fairness_jain": oracle_outcome.fairness_jain,
            "worst_group_utility": oracle_outcome.worst_group_utility,
            "oracle_city_utility": oracle_outcome.city_utility,
            "regret": 0.0,
        })

        # Agent logs
        for jrec in judge_recs:
            agent_rows.append({
                "round_id": i,
                "agent_id": jrec.agent_id,
                "class_id": jrec.class_id,
                "role": "judge",
                "corrupted": jrec.corrupted,
                "adversary_type": jrec.adversary_type.value,
                "recommended_action_id": jrec.action_id,
                "oracle_action_id": oracle_action_id,
            })

        for agg_name, agg in aggregators.items():
            weight_history[agg_name].append(agg.get_weights().copy())

    # Post-hoc oracle_upper_bound
    for agg_name, agg in aggregators.items():
        if isinstance(agg, OracleUpperBoundAggregator):
            retro = agg.retrospective_decisions()
            ub_rows = [r for r in agg_rows if r["aggregator"] == "oracle_upper_bound"]
            for j, row in enumerate(ub_rows):
                if j < len(retro):
                    crisis = crises[j] if j < len(crises) else crises[-1]
                    outcome = world.evaluate_action(crisis, retro[j])
                    row["chosen_action_id"] = retro[j]
                    row["city_utility"] = outcome.city_utility
                    row["unfairness_gap"] = outcome.unfairness
                    row["fairness_jain"] = outcome.fairness_jain
                    row["worst_group_utility"] = outcome.worst_group_utility

    return SimulationResult(
        config=config,
        rounds=[],
        aggregator_log=pd.DataFrame(agg_rows),
        agent_log=pd.DataFrame(agent_rows),
        weight_history=weight_history,
    )
