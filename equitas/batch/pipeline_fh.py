"""
3-stage OpenAI Batch API pipeline for Full-Hierarchy corruption sweeps.

Differences from Governor-Only pipeline (pipeline.py):
  Stage 1: Members — identical (shared across aggregators)
  Stage 2: Each configured aggregator independently runs intra-class selection;
           unique (class, action) pairs get leader prompts (deduped)
  Stage 3: Unique proposal sets get judge prompts (deduped across aggregators)

Output: Full-hierarchy JSONL recordings for replay_fh.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ..aggregators.base import BaseAggregator
from ..aggregators.registry import create_aggregator
from ..agents.parsing import extract_rationale, parse_action_id
from ..agents.prompts import judge_honest_prompt, leader_prompt
from ..config import ExperimentConfig, save_config
from ..metrics.welfare import combined_loss
from ..simulation.replay import save_recording
from ..utils import ensure_dir
from .pipeline import BatchSweepPipeline, LeaderResult, JudgeResult


def _proposal_set_key(choices: Dict[str, str]) -> str:
    """Deterministic key for a set of per-class action choices."""
    return "|".join(f"{cls}:{act}" for cls, act in sorted(choices.items()))


class FHBatchSweepPipeline(BatchSweepPipeline):
    """Full-Hierarchy 3-stage batch pipeline.

    Stage 1: Members (inherited — shared across aggregators)
    Stage 2: Per-aggregator intra-class selection → deduped leader prompts
    Stage 3: Deduped by unique proposal set → judge prompts

    Assembly produces full_hierarchy format recordings for replay_fh.py.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        # FH-specific storage
        # (cond_idx, round_id, cls_id, action_id) → LeaderResult
        self.fh_leader_results: Dict[Tuple[int, int, str, str], LeaderResult] = {}
        # (cond_idx, round_id, pset_key) → List[JudgeResult]
        self.fh_judge_results: Dict[Tuple[int, int, str], List[JudgeResult]] = {}
        # (cond_idx, round_id, agg_name) → {cls_id: action_id}
        self.fh_agg_choices: Dict[Tuple[int, int, str], Dict[str, str]] = {}
        # custom_id decode maps (populated during prepare, used during process)
        self.fh_leader_pair_map: Dict[str, Tuple[int, int, str, str]] = {}
        self.fh_pset_map: Dict[str, Tuple[int, int, str]] = {}

    # -------------------------------------------------------------------
    # Stage 2: Leaders (Full-Hierarchy)
    # -------------------------------------------------------------------

    def stage2_prepare(self) -> str:
        """Per-aggregator intra-class selection → deduped leader prompts."""
        path = os.path.join(str(self.batch_dir), "stage2_leaders.jsonl")
        count = 0
        self.fh_agg_choices = {}
        self.fh_leader_pair_map = {}

        with open(path, "w") as f:
            for cond in self.conditions:
                # Create per-aggregator intra-class aggregators (stateful across rounds)
                per_agg_intra: Dict[str, Dict[str, BaseAggregator]] = {}
                agg_cfgs: Dict[str, Any] = {}
                for agg_cfg in self.config.aggregators:
                    per_agg_intra[agg_cfg.method] = {}
                    agg_cfgs[agg_cfg.method] = agg_cfg
                    for cls_id in self.config.committee.class_ids:
                        per_agg_intra[agg_cfg.method][cls_id] = create_aggregator(
                            agg_cfg,
                            num_agents=self.config.committee.members_per_class,
                        )

                for round_id, crisis in enumerate(cond.crises):
                    key = (cond.idx, round_id)
                    oracle_action_id, oracle_outcome = self.world.best_action(crisis)

                    # --- Per-aggregator intra-class selection ---
                    unique_pairs: Set[Tuple[str, str]] = set()
                    for agg_name, cls_aggs in per_agg_intra.items():
                        choices: Dict[str, str] = {}
                        for cls_id, agg in cls_aggs.items():
                            members = self.member_results[key][cls_id]
                            action_ids = [m.action_id for m in members]
                            chosen = agg.select(action_ids, round_id)
                            choices[cls_id] = chosen
                            unique_pairs.add((cls_id, chosen))
                        self.fh_agg_choices[(cond.idx, round_id, agg_name)] = choices

                    # --- Write leader prompt for each unique (cls, action) pair ---
                    for pair_idx, (cls_id, action_id) in enumerate(sorted(unique_pairs)):
                        members = self.member_results[key][cls_id]
                        member_action_ids = [m.action_id for m in members]
                        vote_counts = Counter(member_action_ids)
                        summary_lines = []
                        for aid, cnt in vote_counts.most_common():
                            summary_lines.append(f"Action {aid}: {cnt} votes")
                        member_summary = "\n".join(summary_lines)

                        crisis_text = self.world.format_round_for_prompt(crisis)
                        sys_msg, usr_msg = leader_prompt(
                            cls_id, crisis_text, member_summary, action_id,
                        )

                        custom_id = f"{cond.idx}_{round_id}_fhl_{pair_idx}"
                        self.fh_leader_pair_map[custom_id] = (
                            cond.idx, round_id, cls_id, action_id,
                        )

                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.config.llm.model,
                                "messages": [
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": usr_msg},
                                ],
                                "temperature": self.config.llm.temperature,
                                "max_tokens": self.config.llm.max_tokens,
                            },
                        }
                        f.write(json.dumps(req) + "\n")
                        count += 1

                    # --- Update all intra-class aggregators with member losses ---
                    for agg_name, cls_aggs in per_agg_intra.items():
                        agg_cfg = agg_cfgs[agg_name]
                        alpha = agg_cfg.alpha if agg_cfg else 1.0
                        beta = agg_cfg.beta if agg_cfg else 0.5
                        for cls_id, agg in cls_aggs.items():
                            members = self.member_results[key][cls_id]
                            member_losses = []
                            for m in members:
                                m_outcome = self.world.evaluate_action(
                                    crisis, m.action_id,
                                )
                                loss = combined_loss(
                                    m_outcome.city_utility,
                                    oracle_outcome.city_utility,
                                    m_outcome.fairness_jain,
                                    alpha=alpha,
                                    beta=beta,
                                )
                                member_losses.append(loss)
                            agg.update(member_losses, round_id)

        print(f"Stage 2 (FH): {count} leader prompts written to {path}")
        return path

    def stage2_process(self, results_path: str) -> None:
        """Parse FH leader batch results."""
        responses = self._load_batch_results(results_path)

        for custom_id, content in responses.items():
            if custom_id not in self.fh_leader_pair_map:
                print(f"  WARNING: unknown leader custom_id {custom_id}, skipping")
                continue

            cond_idx, round_id, cls_id, action_id = self.fh_leader_pair_map[custom_id]

            self.fh_leader_results[(cond_idx, round_id, cls_id, action_id)] = LeaderResult(
                agent_id=f"{cls_id}_leader",
                class_id=cls_id,
                action_id=action_id,
                rationale=content,
            )

        print(f"Stage 2 (FH): processed {len(responses)} leader results")

    # -------------------------------------------------------------------
    # Stage 3: Judges (Full-Hierarchy)
    # -------------------------------------------------------------------

    def stage3_prepare(self) -> str:
        """Deduped judge prompts per unique proposal set."""
        path = os.path.join(str(self.batch_dir), "stage3_judges.jsonl")
        count = 0
        class_ids = self.config.committee.class_ids
        self.fh_pset_map = {}

        with open(path, "w") as f:
            for cond in self.conditions:
                for round_id, crisis in enumerate(cond.crises):
                    crisis_text = self.world.format_round_for_prompt(crisis)

                    # Collect unique proposal sets from all aggregator choices
                    unique_psets: Dict[str, Dict[str, str]] = {}
                    for agg_cfg in self.config.aggregators:
                        agg_name = agg_cfg.method
                        choices = self.fh_agg_choices.get(
                            (cond.idx, round_id, agg_name),
                        )
                        if choices is None:
                            continue
                        pkey = _proposal_set_key(choices)
                        if pkey not in unique_psets:
                            unique_psets[pkey] = choices

                    for pset_idx, pkey in enumerate(sorted(unique_psets)):
                        pset_choices = unique_psets[pkey]
                        result_key = (cond.idx, round_id, pkey)

                        if result_key not in self.fh_judge_results:
                            self.fh_judge_results[result_key] = []

                        # Build proposals text from leader rationales
                        proposals_text: Dict[str, str] = {}
                        for cls_id, action_id in pset_choices.items():
                            lr = self.fh_leader_results.get(
                                (cond.idx, round_id, cls_id, action_id),
                            )
                            if lr:
                                proposals_text[cls_id] = lr.rationale
                            else:
                                proposals_text[cls_id] = "No proposal available."

                        for j_idx in range(self.config.committee.num_judges):
                            is_corrupt = bool(cond.corrupt_judge_mask[j_idx])

                            if is_corrupt and cond.rate > 0:
                                # Algorithmically corrupted judge
                                cls_id = class_ids[j_idx % len(class_ids)]
                                adv = self._make_adversary(
                                    cond.adv_type, cls_id, cond,
                                )
                                valid_ids = [a.id for a in crisis.actions]
                                action_id = adv.corrupt_recommendation(
                                    crisis=crisis,
                                    class_id=cls_id,
                                    honest_action_id=valid_ids[0],
                                    round_id=round_id,
                                )
                                rationale = ""
                                if hasattr(adv, "corrupt_rationale"):
                                    r = adv.corrupt_rationale(
                                        crisis, action_id, cls_id,
                                    )
                                    if r:
                                        rationale = r
                                if not rationale:
                                    rationale = (
                                        f"Selected Action {action_id} as optimal."
                                    )

                                self.fh_judge_results[result_key].append(
                                    JudgeResult(
                                        agent_id=f"judge_{j_idx}",
                                        class_id=cls_id,
                                        action_id=action_id,
                                        rationale=rationale,
                                        corrupted=True,
                                        adversary_type=cond.adv_type,
                                    )
                                )
                            else:
                                # Honest judge — write batch prompt
                                sys_msg, usr_msg = judge_honest_prompt(
                                    crisis_text,
                                    proposals_text,
                                    sees_rationales=True,
                                )
                                custom_id = (
                                    f"{cond.idx}_{round_id}_fhj_{pset_idx}_{j_idx}"
                                )
                                self.fh_pset_map[custom_id] = (
                                    cond.idx, round_id, pkey,
                                )

                                req = {
                                    "custom_id": custom_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": self.config.llm.model,
                                        "messages": [
                                            {"role": "system", "content": sys_msg},
                                            {"role": "user", "content": usr_msg},
                                        ],
                                        "temperature": self.config.llm.temperature,
                                        "max_tokens": self.config.llm.max_tokens,
                                    },
                                }
                                f.write(json.dumps(req) + "\n")
                                count += 1

        print(f"Stage 3 (FH): {count} judge prompts written to {path}")
        return path

    def stage3_process(self, results_path: str) -> None:
        """Parse FH judge batch results."""
        responses = self._load_batch_results(results_path)
        class_ids = self.config.committee.class_ids

        for custom_id, content in responses.items():
            if custom_id not in self.fh_pset_map:
                print(f"  WARNING: unknown judge custom_id {custom_id}, skipping")
                continue

            cond_idx, round_id, pkey = self.fh_pset_map[custom_id]

            # Extract j_idx from custom_id: "{cond}_{round}_fhj_{pset_idx}_{j_idx}"
            parts = custom_id.split("_")
            j_idx = int(parts[-1])

            cond = self.conditions[cond_idx]
            crisis = cond.crises[round_id]
            valid_ids = [a.id for a in crisis.actions]
            cls_id = class_ids[j_idx % len(class_ids)]

            action_id = parse_action_id(content, valid_ids)
            rationale = extract_rationale(content)

            result_key = (cond_idx, round_id, pkey)
            if result_key not in self.fh_judge_results:
                self.fh_judge_results[result_key] = []

            self.fh_judge_results[result_key].append(JudgeResult(
                agent_id=f"judge_{j_idx}",
                class_id=cls_id,
                action_id=action_id,
                rationale=rationale,
                corrupted=False,
                adversary_type="none",
            ))

        # Sort judges by index within each result set
        for key in self.fh_judge_results:
            self.fh_judge_results[key].sort(
                key=lambda r: int(r.agent_id.split("_")[-1]),
            )

        print(f"Stage 3 (FH): processed {len(responses)} judge results")

    # -------------------------------------------------------------------
    # Assembly: Full-Hierarchy recording format
    # -------------------------------------------------------------------

    def assemble_recordings(self) -> None:
        """Write full_hierarchy JSONL recordings, one per condition."""
        rec_dir = os.path.join(str(self.out_dir), "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        class_ids = self.config.committee.class_ids

        for cond in self.conditions:
            rounds_data: List[Dict[str, Any]] = []
            for round_id, crisis in enumerate(cond.crises):
                key = (cond.idx, round_id)
                oracle_action_id, _ = self.world.best_action(crisis)

                # Member recs (per-class, shared across aggregators)
                member_recs: Dict[str, List[Dict[str, Any]]] = {}
                for cls_id in class_ids:
                    members = self.member_results.get(key, {}).get(cls_id, [])
                    member_recs[cls_id] = [{
                        "agent_id": m.agent_id,
                        "class_id": m.class_id,
                        "action_id": m.action_id,
                        "rationale": m.rationale,
                        "corrupted": m.corrupted,
                        "adversary_type": m.adversary_type,
                    } for m in members]

                # Leader calls: keyed by "cls_id:action_id" (deduped)
                leader_calls: Dict[str, Dict[str, Any]] = {}
                for agg_cfg in self.config.aggregators:
                    choices = self.fh_agg_choices.get(
                        (cond.idx, round_id, agg_cfg.method),
                    )
                    if choices is None:
                        continue
                    for cls_id, action_id in choices.items():
                        cache_key = f"{cls_id}:{action_id}"
                        if cache_key not in leader_calls:
                            lr = self.fh_leader_results.get(
                                (cond.idx, round_id, cls_id, action_id),
                            )
                            if lr:
                                leader_calls[cache_key] = {
                                    "agent_id": lr.agent_id,
                                    "class_id": lr.class_id,
                                    "action_id": lr.action_id,
                                    "rationale": lr.rationale,
                                    "corrupted": False,
                                    "adversary_type": "none",
                                }

                # Judge calls: keyed by proposal set key (deduped)
                judge_calls: Dict[str, List[Dict[str, Any]]] = {}
                for agg_cfg in self.config.aggregators:
                    choices = self.fh_agg_choices.get(
                        (cond.idx, round_id, agg_cfg.method),
                    )
                    if choices is None:
                        continue
                    pkey = _proposal_set_key(choices)
                    if pkey not in judge_calls:
                        judges = self.fh_judge_results.get(
                            (cond.idx, round_id, pkey), [],
                        )
                        judge_calls[pkey] = [{
                            "agent_id": j.agent_id,
                            "class_id": j.class_id,
                            "action_id": j.action_id,
                            "rationale": j.rationale,
                            "corrupted": j.corrupted,
                            "adversary_type": j.adversary_type,
                        } for j in judges]

                rounds_data.append({
                    "round_id": round_id,
                    "crisis_id": crisis.id,
                    "oracle_action_id": oracle_action_id,
                    "member_recs": member_recs,
                    "leader_calls": leader_calls,
                    "judge_calls": judge_calls,
                    "format": "full_hierarchy",
                })

            rec_file = (
                f"rate_{cond.rate:.2f}_adv_{cond.adv_type}_run_{cond.run_idx}.jsonl"
            )
            rec_path = os.path.join(rec_dir, rec_file)
            save_recording(rounds_data, rec_path)

        print(
            f"Assembled {len(self.conditions)} full-hierarchy recordings "
            f"in {rec_dir}/"
        )

    # -------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------

    def run(self, resume: bool = False) -> None:
        """Execute all 3 stages and produce full-hierarchy recordings."""
        total = len(self.conditions)
        n_rounds = self.config.world.num_rounds
        print(f"=== FH Batch Pipeline: {total} conditions × {n_rounds} rounds ===")
        print(f"Model: {self.config.llm.model}")
        if resume:
            print("RESUME MODE: skipping completed stages")
        print()

        # Stage 1: Members (inherited from BatchSweepPipeline)
        s1_results_path = os.path.join(
            str(self.batch_dir), "stage1_members_results.jsonl",
        )
        if resume and self._stage_results_exist(1):
            print("--- Stage 1: Members (CACHED) ---")
            self.stage1_prepare()
            self.stage1_process(s1_results_path)
        else:
            print("--- Stage 1: Members ---")
            s1_path = self.stage1_prepare()
            s1_results_path = self.submit_and_wait(s1_path, "Stage 1")
            self.stage1_process(s1_results_path)

        # Stage 2: Leaders (FH — per-aggregator intra-class selection)
        s2_results_path = os.path.join(
            str(self.batch_dir), "stage2_leaders_results.jsonl",
        )
        if resume and self._stage_results_exist(2):
            print("\n--- Stage 2: Leaders/FH (CACHED) ---")
            self.stage2_prepare()
            self.stage2_process(s2_results_path)
        else:
            print("\n--- Stage 2: Leaders (Full-Hierarchy) ---")
            s2_path = self.stage2_prepare()
            s2_results_path = self.submit_and_wait(s2_path, "Stage 2")
            self.stage2_process(s2_results_path)

        # Stage 3: Judges (FH — per unique proposal set)
        s3_results_path = os.path.join(
            str(self.batch_dir), "stage3_judges_results.jsonl",
        )
        if resume and self._stage_results_exist(3):
            print("\n--- Stage 3: Judges/FH (CACHED) ---")
            self.stage3_prepare()
            self.stage3_process(s3_results_path)
        else:
            print("\n--- Stage 3: Judges (Full-Hierarchy) ---")
            s3_path = self.stage3_prepare()
            s3_results_path = self.submit_and_wait(s3_path, "Stage 3")
            self.stage3_process(s3_results_path)

        # Assemble full-hierarchy recordings
        print("\n--- Assembling full-hierarchy recordings ---")
        self.assemble_recordings()

        save_config(
            self.config,
            os.path.join(str(self.out_dir), "config_used.yaml"),
        )
        print(
            f"\n=== FH Batch pipeline complete. Recordings in {self.out_dir}/ ==="
        )
        print(
            "Run aggregation with: python -m equitas --config <yaml> --mode replay "
            f"--replay-dir {self.out_dir}/recordings/"
        )
