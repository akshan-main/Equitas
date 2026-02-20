"""Reproducibility tests for condition seeding and replay."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from equitas.config import (
    AggregatorConfig,
    CommitteeConfig,
    CorruptionConfig,
    ExperimentConfig,
    LLMConfig,
    WorldConfig,
)
from equitas.experiments.sweep import _recording_filename, run_corruption_sweep
from equitas.types import SimulationResult
from equitas.utils import derive_seed


def _tiny_sweep_config(out_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="seed_test",
        environment="governance",
        seed=42,
        num_runs=1,
        llm=LLMConfig(model="test"),
        world=WorldConfig(num_rounds=1, actions_per_crisis=3),
        committee=CommitteeConfig(members_per_class=1, num_judges=1),
        corruption=CorruptionConfig(
            corruption_rate=0.5,
            adversary_type="selfish",
            corruption_target="members",
        ),
        aggregators=[AggregatorConfig(method="majority_vote")],
        corruption_rates=[0.5],
        adversary_types=["selfish"],
        output_dir=out_dir,
    )


def test_derive_seed_stable_across_processes():
    script = (
        "from equitas.utils import derive_seed; "
        "print(derive_seed(42, 0.5, 'selfish', 0))"
    )
    out1 = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    assert out1 == out2


@pytest.mark.asyncio
async def test_replay_uses_condition_derived_seed(tmp_path, monkeypatch):
    cfg = _tiny_sweep_config(str(tmp_path))

    rec_dir = Path(tmp_path) / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_file = rec_dir / _recording_filename(0.5, "selfish", 0)
    rec_file.write_text("{}\n")

    captured_seeds = []

    async def fake_replay(run_config, world, recording_path):
        captured_seeds.append(run_config.seed)
        agg = pd.DataFrame([{
            "round_id": 0,
            "aggregator": "majority_vote",
            "city_utility": 0.5,
            "fairness_jain": 0.5,
            "worst_group_utility": 0.5,
            "regret": 0.1,
        }])
        agent = pd.DataFrame([{
            "round_id": 0,
            "agent_id": "judge_0",
            "class_id": "guardian",
            "role": "judge",
            "corrupted": False,
            "adversary_type": "none",
            "recommended_action_id": "A",
            "oracle_action_id": "A",
        }])
        return SimulationResult(
            config=run_config,
            rounds=[],
            aggregator_log=agg,
            agent_log=agent,
            weight_history={},
        )

    monkeypatch.setattr("equitas.experiments.sweep.replay_simulation", fake_replay)
    monkeypatch.setattr(
        "equitas.plotting.corruption_sweep.plot_corruption_sweep",
        lambda summary, out_dir: None,
    )
    monkeypatch.setattr(
        "equitas.plotting.regime_plots.plot_winner_bands",
        lambda summary, out_dir: None,
    )
    monkeypatch.setattr(
        "equitas.metrics.regime_analysis.analyze_regimes",
        lambda summary: SimpleNamespace(
            regime_map=pd.DataFrame(),
            regime_map_detailed=pd.DataFrame(),
        ),
    )
    monkeypatch.setattr("equitas.metrics.regime_analysis.format_regime_report", lambda report: "")
    monkeypatch.setattr("equitas.metrics.regime_analysis.format_regime_map_markdown", lambda report: "")

    await run_corruption_sweep(cfg, mode="replay", replay_dir=str(rec_dir))

    assert captured_seeds == [derive_seed(42, 0.5, "selfish", 0)]
