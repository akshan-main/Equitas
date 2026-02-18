"""
Core hierarchical simulation engine.

Per round:
  Level 1 (intra-class): N members -> leader proposal (per class)
  Level 2 (inter-class): 3 proposals -> M judges -> governor decision
  Evaluate + update aggregator weights
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..adversaries.adaptive import ScheduledAdversary
from ..adversaries.base import BaseAdversary
from ..adversaries.coordinated import CoordinatedAdversary
from ..adversaries.corruption_schedule import CorruptionSchedule
from ..adversaries.deceptive import DeceptiveAdversary
from ..adversaries.selfish import SelfishAdversary
from ..agents.governor import Governor
from ..agents.judge import JudgeAgent, create_judges
from ..agents.leader import LeaderAgent
from ..agents.llm_client import LLMClientWrapper
from ..agents.member import MemberAgent, create_class_members
from ..aggregators.base import BaseAggregator
from ..aggregators.best_single import OracleUpperBoundAggregator
from ..aggregators.registry import create_aggregator
from ..config import AggregatorConfig, ExperimentConfig
from ..metrics.fairness import jain_fairness_index, unfairness_gap, worst_group_utility
from ..metrics.welfare import combined_loss, regret
from ..types import (
    AdversaryType,
    AgentRecommendation,
    Crisis,
    Outcome,
    RoundResult,
    SimulationResult,
)
from ..worlds.base import BaseWorld


@dataclass
class HierarchicalCommittee:
    """Complete committee structure for one simulation run."""
    members: Dict[str, List[MemberAgent]]
    leaders: Dict[str, LeaderAgent]
    judges: List[JudgeAgent]
    governor_aggregators: Dict[str, Governor]
    intra_class_aggregators: Dict[str, BaseAggregator]
    # Keep references for best_single post-hoc
    _aggregator_configs: Dict[str, AggregatorConfig]


def _make_adversary_factory(
    adversary_type: str,
    world: BaseWorld,
    llm_client: LLMClientWrapper,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Optional[Callable]:
    """Create a factory that produces adversary instances for a given class."""
    adv = AdversaryType(adversary_type)
    if adv == AdversaryType.NONE:
        return None

    def factory(class_id: str) -> BaseAdversary:
        if adv == AdversaryType.SELFISH:
            return SelfishAdversary(world=world, class_id=class_id)
        elif adv == AdversaryType.COORDINATED:
            return CoordinatedAdversary(
                world=world,
                class_id=class_id,
                target_strategy=config.corruption.coordinated_target,
                rng=rng,
            )
        elif adv == AdversaryType.SCHEDULED:
            return ScheduledAdversary(
                world=world,
                class_id=class_id,
                honest_rounds=config.corruption.scheduled_honest_rounds,
                rng=rng,
            )
        elif adv == AdversaryType.DECEPTIVE:
            return DeceptiveAdversary(
                world=world,
                class_id=class_id,
                llm_client=llm_client,
                strength=config.corruption.deceptive_strength,
                rng=rng,
            )
        else:
            return SelfishAdversary(world=world, class_id=class_id)

    return factory


def _build_committee(
    config: ExperimentConfig,
    world: BaseWorld,
    llm_client: LLMClientWrapper,
    rng: np.random.Generator,
) -> HierarchicalCommittee:
    """Construct the full hierarchical committee from config."""
    class_ids = config.committee.class_ids
    adv_type = config.corruption.adversary_type
    corruption_target = config.corruption.corruption_target

    adversary_factory = _make_adversary_factory(
        adv_type, world, llm_client, config, rng,
    )

    # Determine corruption rates for members vs judges
    member_rate = config.corruption.corruption_rate if corruption_target in ("members", "both") else 0.0
    judge_rate = config.corruption.corruption_rate if corruption_target in ("judges", "both") else 0.0

    # Create intra-class aggregators (trimmed vote for robust within-class aggregation)
    intra_aggs: Dict[str, BaseAggregator] = {}
    for cls_id in class_ids:
        from ..aggregators.trimmed_vote import TrimmedVoteAggregator
        intra_aggs[cls_id] = TrimmedVoteAggregator(
            num_agents=config.committee.members_per_class,
            trim_fraction=0.2,
        )

    # Create members per class
    realization = config.corruption.corruption_realization
    all_members: Dict[str, List[MemberAgent]] = {}
    for cls_id in class_ids:
        all_members[cls_id] = create_class_members(
            class_id=cls_id,
            num_members=config.committee.members_per_class,
            corruption_rate=member_rate,
            adversary_type=AdversaryType(adv_type),
            llm_client=llm_client,
            world=world,
            rng=rng,
            adversary_factory=adversary_factory,
            corruption_realization=realization,
        )

    # Create leaders per class
    leaders: Dict[str, LeaderAgent] = {}
    for cls_id in class_ids:
        leaders[cls_id] = LeaderAgent(
            agent_id=f"{cls_id}_leader",
            class_id=cls_id,
            intra_class_aggregator=intra_aggs[cls_id],
            llm_client=llm_client,
            world=world,
        )

    # Create judges
    judges = create_judges(
        num_judges=config.committee.num_judges,
        class_ids=class_ids,
        corruption_rate=judge_rate,
        adversary_type=AdversaryType(adv_type),
        llm_client=llm_client,
        world=world,
        rng=rng,
        sees_rationales=True,
        adversary_factory=adversary_factory,
    )

    # Create governor with each aggregator method
    governor_aggs: Dict[str, Governor] = {}
    agg_configs: Dict[str, AggregatorConfig] = {}
    for agg_cfg in config.aggregators:
        agg = create_aggregator(agg_cfg, num_agents=config.committee.num_judges)
        governor_aggs[agg_cfg.method] = Governor(aggregator=agg)
        agg_configs[agg_cfg.method] = agg_cfg

    return HierarchicalCommittee(
        members=all_members,
        leaders=leaders,
        judges=judges,
        governor_aggregators=governor_aggs,
        intra_class_aggregators=intra_aggs,
        _aggregator_configs=agg_configs,
    )


def _collect_all_agents(committee: HierarchicalCommittee) -> list:
    """Collect all agents (members + judges) as flat list."""
    agents: list = []
    for cls_members in committee.members.values():
        agents.extend(cls_members)
    agents.extend(committee.judges)
    return agents


def _get_corrupted_ids(committee: HierarchicalCommittee) -> List[str]:
    """Collect IDs of all agents initially marked as corrupted."""
    ids = []
    for cls_members in committee.members.values():
        for m in cls_members:
            if m.corrupted:
                ids.append(m.agent_id)
    for j in committee.judges:
        if j.corrupted:
            ids.append(j.agent_id)
    return ids


async def _run_class_level1(
    cls_id: str,
    round_id: int,
    crisis: Crisis,
    committee: HierarchicalCommittee,
) -> tuple:
    """Run Level 1 for a single class: members recommend, leader aggregates."""
    members = committee.members[cls_id]
    leader = committee.leaders[cls_id]

    # Members recommend in parallel
    member_tasks = [m.act(crisis, round_id) for m in members]
    member_recs: List[AgentRecommendation] = await asyncio.gather(*member_tasks)

    # Leader aggregates
    leader_proposal = await leader.act(crisis, member_recs, round_id)

    return member_recs, leader_proposal


async def _execute_round(
    round_id: int,
    crisis: Crisis,
    committee: HierarchicalCommittee,
    world: BaseWorld,
    config: ExperimentConfig,
) -> RoundResult:
    """Execute one round of hierarchical governance."""
    # Oracle
    oracle_action_id, oracle_outcome = world.best_action(crisis)

    # --- LEVEL 1: Intra-class ---
    class_member_recs: Dict[str, List[AgentRecommendation]] = {}
    class_leader_proposals: Dict[str, AgentRecommendation] = {}

    level1_tasks = [
        _run_class_level1(cls_id, round_id, crisis, committee)
        for cls_id in config.committee.class_ids
    ]
    level1_results = await asyncio.gather(*level1_tasks)

    for cls_id, (member_recs, leader_proposal) in zip(
        config.committee.class_ids, level1_results,
    ):
        class_member_recs[cls_id] = member_recs
        class_leader_proposals[cls_id] = leader_proposal

    # --- LEVEL 2: Inter-class ---
    judge_tasks = [
        judge.act(crisis, class_leader_proposals, round_id)
        for judge in committee.judges
    ]
    judge_evaluations: List[AgentRecommendation] = await asyncio.gather(*judge_tasks)

    # Governor decides for each aggregator
    aggregator_decisions: Dict[str, str] = {}
    aggregator_outcomes: Dict[str, Outcome] = {}

    for agg_name, governor in committee.governor_aggregators.items():
        chosen = governor.decide(judge_evaluations, round_id)
        outcome = world.evaluate_action(crisis, chosen)
        aggregator_decisions[agg_name] = chosen
        aggregator_outcomes[agg_name] = outcome

    # Compute per-judge losses and update each governor
    for agg_name, governor in committee.governor_aggregators.items():
        agg_cfg = committee._aggregator_configs.get(agg_name)
        alpha = agg_cfg.alpha if agg_cfg else 1.0
        beta = agg_cfg.beta if agg_cfg else 0.5

        judge_losses = []
        for jrec in judge_evaluations:
            j_outcome = world.evaluate_action(crisis, jrec.action_id)
            loss = combined_loss(
                j_outcome.city_utility,
                oracle_outcome.city_utility,
                j_outcome.fairness_jain,
                alpha=alpha,
                beta=beta,
            )
            judge_losses.append(loss)
        governor.update(judge_losses, round_id)

    # Update intra-class aggregators for members
    for cls_id in config.committee.class_ids:
        member_recs = class_member_recs[cls_id]
        agg_cfg = config.aggregators[0] if config.aggregators else AggregatorConfig()
        member_losses = []
        for rec in member_recs:
            m_outcome = world.evaluate_action(crisis, rec.action_id)
            loss = combined_loss(
                m_outcome.city_utility,
                oracle_outcome.city_utility,
                m_outcome.fairness_jain,
                alpha=agg_cfg.alpha,
                beta=agg_cfg.beta,
            )
            member_losses.append(loss)
        committee.intra_class_aggregators[cls_id].update(member_losses, round_id)

    # Weight snapshots
    mw_snap = {}
    for agg_name, gov in committee.governor_aggregators.items():
        mw_snap[agg_name] = gov.get_weights().copy()

    return RoundResult(
        round_id=round_id,
        crisis=crisis,
        oracle_action_id=oracle_action_id,
        oracle_outcome=oracle_outcome,
        class_member_recs=class_member_recs,
        class_leader_proposals=class_leader_proposals,
        judge_evaluations=judge_evaluations,
        aggregator_decisions=aggregator_decisions,
        aggregator_outcomes=aggregator_outcomes,
        mw_weights_snapshot=mw_snap,
    )


def _log_round(
    rr: RoundResult,
    config: ExperimentConfig,
    agg_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
) -> None:
    """Append logging rows for this round."""
    # Aggregator log
    for agg_name, outcome in rr.aggregator_outcomes.items():
        agg_rows.append({
            "round_id": rr.round_id,
            "aggregator": agg_name,
            "chosen_action_id": rr.aggregator_decisions[agg_name],
            "city_utility": outcome.city_utility,
            "unfairness_gap": outcome.unfairness,
            "fairness_jain": outcome.fairness_jain,
            "worst_group_utility": outcome.worst_group_utility,
            "oracle_city_utility": rr.oracle_outcome.city_utility,
            "regret": rr.oracle_outcome.city_utility - outcome.city_utility,
        })

    # Oracle row
    agg_rows.append({
        "round_id": rr.round_id,
        "aggregator": "oracle",
        "chosen_action_id": rr.oracle_action_id,
        "city_utility": rr.oracle_outcome.city_utility,
        "unfairness_gap": rr.oracle_outcome.unfairness,
        "fairness_jain": rr.oracle_outcome.fairness_jain,
        "worst_group_utility": rr.oracle_outcome.worst_group_utility,
        "oracle_city_utility": rr.oracle_outcome.city_utility,
        "regret": 0.0,
    })

    # Agent log: members
    for cls_id, recs in rr.class_member_recs.items():
        for rec in recs:
            agent_rows.append({
                "round_id": rr.round_id,
                "agent_id": rec.agent_id,
                "class_id": rec.class_id,
                "role": "member",
                "corrupted": rec.corrupted,
                "adversary_type": rec.adversary_type.value,
                "recommended_action_id": rec.action_id,
                "oracle_action_id": rr.oracle_action_id,
            })

    # Agent log: judges
    for jrec in rr.judge_evaluations:
        agent_rows.append({
            "round_id": rr.round_id,
            "agent_id": jrec.agent_id,
            "class_id": jrec.class_id,
            "role": "judge",
            "corrupted": jrec.corrupted,
            "adversary_type": jrec.adversary_type.value,
            "recommended_action_id": jrec.action_id,
            "oracle_action_id": rr.oracle_action_id,
        })


def _finalize_best_single(
    committee: HierarchicalCommittee,
    rounds: List[RoundResult],
    agg_rows: List[Dict[str, Any]],
    world: BaseWorld,
) -> None:
    """Post-hoc: replace best_single placeholder outcomes with retrospective best."""
    gov = committee.governor_aggregators.get("oracle_upper_bound")
    if gov is None:
        return
    agg = gov.aggregator
    if not isinstance(agg, OracleUpperBoundAggregator):
        return

    retro_decisions = agg.retrospective_decisions()
    # Update the agg_rows that have aggregator="oracle_upper_bound"
    best_single_rows = [r for r in agg_rows if r["aggregator"] == "oracle_upper_bound"]
    for i, row in enumerate(best_single_rows):
        if i < len(retro_decisions):
            crisis = rounds[i].crisis
            new_action = retro_decisions[i]
            outcome = world.evaluate_action(crisis, new_action)
            row["chosen_action_id"] = new_action
            row["city_utility"] = outcome.city_utility
            row["unfairness_gap"] = outcome.unfairness
            row["fairness_jain"] = outcome.fairness_jain
            row["worst_group_utility"] = outcome.worst_group_utility
            row["regret"] = rounds[i].oracle_outcome.city_utility - outcome.city_utility


async def run_hierarchical_simulation(
    config: ExperimentConfig,
    world: BaseWorld,
    seed: int,
    recording_path: Optional[str] = None,
) -> SimulationResult:
    """
    Main entry point for one simulation run.

    If recording_path is set, saves all agent recommendations to JSONL
    so the run can be replayed with different aggregators (zero API calls).
    """
    from .replay import rec_from_agent_rec, record_round, save_recording

    rng = np.random.default_rng(seed)
    llm_client = LLMClientWrapper(config.llm)

    # Build committee
    committee = _build_committee(config, world, llm_client, rng)

    # Generate rounds
    crises = world.generate_rounds(config.world.num_rounds, rng)

    # Corruption schedule
    corrupted_ids = _get_corrupted_ids(committee)
    schedule = CorruptionSchedule(
        onset_round=config.corruption.corruption_onset_round or 0,
        corrupted_agent_ids=corrupted_ids,
    )

    # Run simulation loop
    round_results: List[RoundResult] = []
    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    weight_history: Dict[str, List[np.ndarray]] = {
        agg_name: [] for agg_name in committee.governor_aggregators
    }
    recording_data: List[Dict[str, Any]] = []

    for round_id, crisis in enumerate(crises):
        # Apply corruption schedule
        all_agents = _collect_all_agents(committee)
        schedule.apply_to_agents(all_agents, round_id)

        # Execute round
        print(f"  Round {round_id + 1}/{len(crises)}", end="\r")
        rr = await _execute_round(round_id, crisis, committee, world, config)
        round_results.append(rr)

        # Log
        _log_round(rr, config, agg_rows, agent_rows)

        # Record for replay
        if recording_path is not None:
            m_ser: Dict[str, list] = {}
            for cls_id, recs in rr.class_member_recs.items():
                m_ser[cls_id] = [rec_from_agent_rec(r) for r in recs]
            l_ser: Dict[str, dict] = {}
            for cls_id, prop in rr.class_leader_proposals.items():
                l_ser[cls_id] = rec_from_agent_rec(prop)
            j_ser = [rec_from_agent_rec(j) for j in rr.judge_evaluations]
            recording_data.append(record_round(
                round_id=round_id, crisis_id=crisis.id,
                member_recs=m_ser, leader_proposals=l_ser,
                judge_recs=j_ser, oracle_action_id=rr.oracle_action_id,
            ))

        # Weight snapshots
        for agg_name, gov in committee.governor_aggregators.items():
            weight_history[agg_name].append(gov.get_weights().copy())

    # Post-hoc best_single
    _finalize_best_single(committee, round_results, agg_rows, world)

    # Save recording
    if recording_path is not None and recording_data:
        save_recording(recording_data, recording_path)
        print(f"\n  Recording saved to {recording_path}")

    print()  # newline after progress

    return SimulationResult(
        config=config,
        rounds=round_results,
        aggregator_log=pd.DataFrame(agg_rows),
        agent_log=pd.DataFrame(agent_rows),
        weight_history=weight_history,
    )
