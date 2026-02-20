"""Tests for the OpenAI Batch API pipeline.

Tests both structural correctness (JSONL format, recording schema)
and end-to-end execution with mocked batch API responses.
"""
from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from equitas.batch.pipeline import BatchSweepPipeline, MemberResult
from equitas.config import ExperimentConfig, LLMConfig, WorldConfig, CommitteeConfig, CorruptionConfig, AggregatorConfig
from equitas.metrics.welfare import combined_loss
from equitas.simulation.replay import load_recording, agent_rec_from_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(tmp_dir: str, corruption_realization: str = "algorithmic",
                 adversary_types: list = None, corruption_rates: list = None) -> ExperimentConfig:
    """Create minimal config for testing: 1 rate × 1 adv × 1 run × 2 rounds."""
    return ExperimentConfig(
        name="batch_test",
        environment="governance",
        seed=42,
        num_runs=1,
        llm=LLMConfig(model="gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
        world=WorldConfig(
            crisis_axes=["resource_scarcity", "external_threat"],
            policy_dims=["tax_merchants", "welfare_workers"],
            actions_per_crisis=3,
            num_rounds=2,
        ),
        committee=CommitteeConfig(
            class_ids=["guardian", "producer"],
            members_per_class=3,
            num_judges=2,
        ),
        corruption=CorruptionConfig(
            corruption_rate=0.25,
            adversary_type="selfish",
            corruption_realization=corruption_realization,
            corruption_target="members",
            coordinated_target="worst_city",
            scheduled_honest_rounds=1,
            deceptive_strength="strong",
        ),
        aggregators=[AggregatorConfig(method="majority_vote")],
        corruption_rates=corruption_rates or [0.0, 0.5],
        adversary_types=adversary_types or ["selfish"],
        output_dir=tmp_dir,
    )


def _fake_batch_response(custom_id: str, action_letter: str = "A") -> str:
    """Create a fake OpenAI Batch API response line."""
    return json.dumps({
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{
                    "message": {
                        "content": (
                            f"After careful analysis, Action {action_letter} is best "
                            "for overall city welfare.\n\n"
                            f"ACTION_ID: {action_letter}"
                        ),
                    }
                }],
            },
        },
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConditionPreparation:
    """Test that conditions are deterministically generated."""

    def test_condition_count(self, tmp_path):
        cfg = _tiny_config(str(tmp_path), adversary_types=["selfish"], corruption_rates=[0.0, 0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        # 2 rates × 1 adv × 1 run = 2 conditions
        assert len(pipeline.conditions) == 2

    def test_crises_deterministic(self, tmp_path):
        """Same config → same crises."""
        cfg = _tiny_config(str(tmp_path))
        os.environ["OPENAI_API_KEY"] = "test-key"
        p1 = BatchSweepPipeline(cfg)
        p2 = BatchSweepPipeline(cfg)
        for c1, c2 in zip(p1.conditions, p2.conditions):
            assert c1.seed == c2.seed
            assert len(c1.crises) == len(c2.crises)
            for cr1, cr2 in zip(c1.crises, c2.crises):
                assert cr1.axes == cr2.axes
                for a1, a2 in zip(cr1.actions, cr2.actions):
                    assert a1.id == a2.id
                    assert a1.policy == a2.policy

    def test_corruption_masks(self, tmp_path):
        cfg = _tiny_config(str(tmp_path), corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        cond = pipeline.conditions[0]
        # 50% of 3 members = 1 corrupted per class
        for cls_id in cfg.committee.class_ids:
            assert cond.corrupt_member_mask[cls_id].sum() == 1


class TestStage1Members:
    """Test member prompt generation and algorithmic member computation."""

    def test_algorithmic_selfish_no_llm(self, tmp_path):
        """Selfish + algorithmic realization → corrupted members skip LLM."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        cond = pipeline.conditions[0]
        for cls_id in cfg.committee.class_ids:
            for m_idx in range(cfg.committee.members_per_class):
                needs_llm = pipeline._member_needs_llm(cond, cls_id, m_idx, round_id=0)
                is_corrupt = bool(cond.corrupt_member_mask[cls_id][m_idx])
                if is_corrupt:
                    assert not needs_llm, f"Selfish+algorithmic corrupt member should NOT need LLM"
                else:
                    assert needs_llm, "Honest member always needs LLM"

    def test_llm_realization_always_needs_llm(self, tmp_path):
        """LLM realization → ALL members need LLM (honest + corrupt)."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="llm",
                          adversary_types=["selfish"], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        cond = pipeline.conditions[0]
        for cls_id in cfg.committee.class_ids:
            for m_idx in range(cfg.committee.members_per_class):
                assert pipeline._member_needs_llm(cond, cls_id, m_idx, round_id=0)

    def test_stage1_prepare_writes_valid_jsonl(self, tmp_path):
        """Stage1 produces valid JSONL with correct batch API format."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.0])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        path = pipeline.stage1_prepare()

        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # 0% corruption, 2 classes × 3 members × 2 rounds = 12 prompts
        assert len(lines) == 12

        for req in lines:
            assert "custom_id" in req
            assert req["method"] == "POST"
            assert req["url"] == "/v1/chat/completions"
            assert "messages" in req["body"]
            assert req["body"]["model"] == "gpt-4o-mini"
            msgs = req["body"]["messages"]
            assert len(msgs) == 2
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    def test_stage1_prepare_fewer_prompts_with_corruption(self, tmp_path):
        """With algorithmic corruption, corrupted members are pre-computed, not in JSONL."""
        cfg_clean = _tiny_config(str(tmp_path / "clean"), corruption_realization="algorithmic",
                                adversary_types=["selfish"], corruption_rates=[0.0])
        cfg_corrupt = _tiny_config(str(tmp_path / "corrupt"), corruption_realization="algorithmic",
                                  adversary_types=["selfish"], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"

        p_clean = BatchSweepPipeline(cfg_clean)
        p_corrupt = BatchSweepPipeline(cfg_corrupt)

        path_clean = p_clean.stage1_prepare()
        path_corrupt = p_corrupt.stage1_prepare()

        with open(path_clean) as f:
            n_clean = sum(1 for line in f if line.strip())
        with open(path_corrupt) as f:
            n_corrupt = sum(1 for line in f if line.strip())

        # Corrupt pipeline should have fewer LLM prompts (some pre-computed)
        assert n_corrupt < n_clean

    def test_algorithmic_member_produces_valid_action(self, tmp_path):
        """Algorithmically computed corrupted members select valid actions."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        cond = pipeline.conditions[0]
        crisis = cond.crises[0]
        valid_ids = [a.id for a in crisis.actions]

        for cls_id in cfg.committee.class_ids:
            for m_idx in range(cfg.committee.members_per_class):
                if cond.corrupt_member_mask[cls_id][m_idx]:
                    result = pipeline._compute_algorithmic_member(
                        cond, cls_id, m_idx, crisis, round_id=0,
                    )
                    assert result.action_id in valid_ids
                    assert result.corrupted is True
                    assert result.adversary_type == "selfish"


class TestStage1Process:
    """Test parsing of batch API results for members."""

    def test_stage1_process_parses_correctly(self, tmp_path):
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.0])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        pipeline.stage1_prepare()

        # Create fake results file
        results_path = str(tmp_path / "stage1_results.jsonl")
        with open(results_path, "w") as f:
            for cond in pipeline.conditions:
                for round_id in range(cfg.world.num_rounds):
                    for cls_id in cfg.committee.class_ids:
                        for m_idx in range(cfg.committee.members_per_class):
                            cid = f"{cond.idx}_{round_id}_member_{cls_id}_{m_idx}"
                            f.write(_fake_batch_response(cid, "B") + "\n")

        pipeline.stage1_process(results_path)

        # Verify results populated
        for cond in pipeline.conditions:
            for round_id in range(cfg.world.num_rounds):
                key = (cond.idx, round_id)
                assert key in pipeline.member_results
                for cls_id in cfg.committee.class_ids:
                    members = pipeline.member_results[key][cls_id]
                    assert len(members) == cfg.committee.members_per_class
                    for m in members:
                        assert m.action_id == "B"
                        assert m.class_id == cls_id


class TestStage2StatefulSelection:
    """Stage 2 should mirror stateful intra-class aggregation from engine.py."""

    def test_stage2_uses_stateful_trimmed_vote(self, tmp_path):
        cfg = ExperimentConfig(
            name="batch_stage2_stateful",
            environment="governance",
            seed=123,
            num_runs=1,
            llm=LLMConfig(model="gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
            world=WorldConfig(
                crisis_axes=["resource_scarcity", "external_threat"],
                policy_dims=["tax_merchants", "welfare_workers"],
                actions_per_crisis=3,
                num_rounds=2,
            ),
            committee=CommitteeConfig(
                class_ids=["guardian"],
                members_per_class=5,  # trim_fraction=0.2 => trims 1 member
                num_judges=1,
            ),
            corruption=CorruptionConfig(
                corruption_rate=0.0,
                adversary_type="selfish",
                corruption_realization="algorithmic",
                corruption_target="members",
            ),
            aggregators=[AggregatorConfig(method="majority_vote", alpha=1.0, beta=0.5)],
            corruption_rates=[0.0],
            adversary_types=["selfish"],
            output_dir=str(tmp_path),
        )
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        cond = pipeline.conditions[0]

        # Build round 0 so member_0 has the highest cumulative loss.
        crisis0 = cond.crises[0]
        action_ids0 = [a.id for a in crisis0.actions]
        _, oracle_outcome0 = pipeline.world.best_action(crisis0)

        loss_by_action = {}
        for aid in action_ids0:
            out = pipeline.world.evaluate_action(crisis0, aid)
            loss_by_action[aid] = combined_loss(
                out.city_utility,
                oracle_outcome0.city_utility,
                out.fairness_jain,
                alpha=1.0,
                beta=0.5,
            )
        bad_action = max(loss_by_action, key=loss_by_action.get)
        good_action = min(loss_by_action, key=loss_by_action.get)

        # Round 1 votes are set so that removing member_0 flips winner A -> B.
        crisis1 = cond.crises[1]
        a_id, b_id, c_id = sorted([a.id for a in crisis1.actions])

        def _m(i: int, action_id: str) -> MemberResult:
            return MemberResult(
                agent_id=f"guardian_member_{i}",
                class_id="guardian",
                action_id=action_id,
                rationale="",
                corrupted=False,
                adversary_type="none",
            )

        pipeline.member_results[(cond.idx, 0)] = {
            "guardian": [
                _m(0, bad_action),
                _m(1, good_action),
                _m(2, good_action),
                _m(3, good_action),
                _m(4, good_action),
            ],
        }
        pipeline.member_results[(cond.idx, 1)] = {
            "guardian": [
                _m(0, a_id),
                _m(1, a_id),
                _m(2, b_id),
                _m(3, b_id),
                _m(4, c_id),
            ],
        }

        stage2_path = pipeline.stage2_prepare()
        assert os.path.exists(stage2_path)

        # If stage2 were stateless, round 1 winner would be A (tie-break A vs B).
        # Stateful trimmed vote should trim member_0 and pick B.
        picked = pipeline.leader_selected_actions[(cond.idx, 1, "guardian")]
        assert picked == b_id

        with open(stage2_path) as f:
            reqs = [json.loads(line) for line in f if line.strip()]
        round1_req = next(r for r in reqs if r["custom_id"] == f"{cond.idx}_1_leader_guardian")
        system_msg = round1_req["body"]["messages"][0]["content"]
        assert f"Action {b_id}" in system_msg


class TestEndToEnd:
    """Full pipeline test with mocked Batch API."""

    def test_full_pipeline_produces_recordings(self, tmp_path):
        """Run all 3 stages with mock API and verify recording output."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.0])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)

        # Mock submit_and_wait to generate fake results from the JSONL files
        original_prepare_s1 = pipeline.stage1_prepare
        original_prepare_s2 = pipeline.stage2_prepare
        original_prepare_s3 = pipeline.stage3_prepare

        def _mock_submit_and_wait(jsonl_path: str, stage_name: str) -> str:
            """Read the request JSONL, generate mock responses."""
            results_path = jsonl_path.replace(".jsonl", "_results.jsonl")
            with open(jsonl_path) as fin, open(results_path, "w") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    cid = req["custom_id"]
                    fout.write(_fake_batch_response(cid, "A") + "\n")
            return results_path

        pipeline.submit_and_wait = _mock_submit_and_wait

        # Run the full pipeline
        pipeline.run()

        # Check recordings were created
        rec_dir = tmp_path / "recordings"
        assert rec_dir.exists()

        rec_files = list(rec_dir.glob("*.jsonl"))
        assert len(rec_files) == 1  # 1 rate × 1 adv × 1 run = 1 recording

        # Validate recording structure
        recording = load_recording(str(rec_files[0]))
        assert len(recording) == 2  # 2 rounds

        for rec in recording:
            assert "round_id" in rec
            assert "crisis_id" in rec
            assert "oracle_action_id" in rec
            assert "member_recs" in rec
            assert "leader_proposals" in rec
            assert "judge_recs" in rec

            # Check member structure
            for cls_id in cfg.committee.class_ids:
                assert cls_id in rec["member_recs"]
                members = rec["member_recs"][cls_id]
                assert len(members) == cfg.committee.members_per_class
                for m in members:
                    assert "agent_id" in m
                    assert "class_id" in m
                    assert "action_id" in m
                    assert "rationale" in m
                    assert "corrupted" in m
                    assert "adversary_type" in m

            # Check leader structure
            for cls_id in cfg.committee.class_ids:
                assert cls_id in rec["leader_proposals"]
                lp = rec["leader_proposals"][cls_id]
                assert "agent_id" in lp
                assert "action_id" in lp
                assert "rationale" in lp

            # Check judge structure
            assert len(rec["judge_recs"]) == cfg.committee.num_judges
            for j in rec["judge_recs"]:
                assert "agent_id" in j
                assert "action_id" in j
                assert "corrupted" in j

    def test_recording_loadable_by_replay(self, tmp_path):
        """Recordings from batch pipeline can be loaded by replay system."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.0])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)

        def _mock_submit_and_wait(jsonl_path: str, stage_name: str) -> str:
            results_path = jsonl_path.replace(".jsonl", "_results.jsonl")
            with open(jsonl_path) as fin, open(results_path, "w") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    cid = req["custom_id"]
                    fout.write(_fake_batch_response(cid, "A") + "\n")
            return results_path

        pipeline.submit_and_wait = _mock_submit_and_wait
        pipeline.run()

        # Load recording and convert to AgentRecommendation objects
        rec_dir = tmp_path / "recordings"
        rec_files = list(rec_dir.glob("*.jsonl"))
        recording = load_recording(str(rec_files[0]))

        for rec in recording:
            # Judge recs should be convertible
            for j in rec["judge_recs"]:
                agent_rec = agent_rec_from_dict(j)
                assert agent_rec.agent_id.startswith("judge_")

    def test_corrupt_condition_has_corrupt_members(self, tmp_path):
        """Recordings for corrupted conditions show corrupted=True members."""
        cfg = _tiny_config(str(tmp_path), corruption_realization="algorithmic",
                          adversary_types=["selfish"], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)

        def _mock_submit_and_wait(jsonl_path: str, stage_name: str) -> str:
            results_path = jsonl_path.replace(".jsonl", "_results.jsonl")
            with open(jsonl_path) as fin, open(results_path, "w") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    cid = req["custom_id"]
                    fout.write(_fake_batch_response(cid, "A") + "\n")
            return results_path

        pipeline.submit_and_wait = _mock_submit_and_wait
        pipeline.run()

        rec_dir = tmp_path / "recordings"
        rec_files = list(rec_dir.glob("rate_0.50*.jsonl"))
        assert len(rec_files) == 1

        recording = load_recording(str(rec_files[0]))
        # Check at least one member is marked corrupted
        has_corrupt = False
        for rec in recording:
            for cls_id in rec["member_recs"]:
                for m in rec["member_recs"][cls_id]:
                    if m["corrupted"]:
                        has_corrupt = True
                        assert m["adversary_type"] == "selfish"
        assert has_corrupt, "50% corruption should produce at least one corrupted member"


class TestAllAdversaryTypes:
    """Test pipeline handles all 4 adversary types."""

    @pytest.mark.parametrize("adv_type", ["selfish", "coordinated", "scheduled", "deceptive"])
    def test_adversary_type_stage1(self, tmp_path, adv_type):
        """Stage 1 prepares correctly for each adversary type."""
        cfg = _tiny_config(str(tmp_path / adv_type), corruption_realization="algorithmic",
                          adversary_types=[adv_type], corruption_rates=[0.5])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)
        path = pipeline.stage1_prepare()

        # Should not crash, and should produce a valid JSONL
        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        # At least some LLM prompts (honest members always need LLM)
        assert len(lines) > 0


class TestConfigSaved:
    """Test that config is saved alongside recordings."""

    def test_config_saved(self, tmp_path):
        cfg = _tiny_config(str(tmp_path), adversary_types=["selfish"], corruption_rates=[0.0])
        os.environ["OPENAI_API_KEY"] = "test-key"
        pipeline = BatchSweepPipeline(cfg)

        def _mock_submit_and_wait(jsonl_path: str, stage_name: str) -> str:
            results_path = jsonl_path.replace(".jsonl", "_results.jsonl")
            with open(jsonl_path) as fin, open(results_path, "w") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    cid = req["custom_id"]
                    fout.write(_fake_batch_response(cid, "A") + "\n")
            return results_path

        pipeline.submit_and_wait = _mock_submit_and_wait
        pipeline.run()

        assert (tmp_path / "config_used.yaml").exists()
