"""
Corruption rate sweep experiment.
corruption_rate x adversary_type x aggregation_method

Supports record/replay:
  record mode: run LLMs, save JSONL recordings per condition
  replay mode: re-run aggregation from saved recordings (zero API calls)
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ExperimentConfig, save_config
from ..metrics.robust_stats import bootstrap_ci, trimmed_mean
from ..simulation.engine import run_hierarchical_simulation
from ..simulation.replay import replay_simulation
from ..utils import derive_seed, ensure_dir
from ..worlds.governance import GovernanceWorld
from ..worlds.base import BaseWorld


def _create_world(config: ExperimentConfig) -> BaseWorld:
    if config.environment == "gsm8k":
        from ..worlds.gsm8k import GSM8KWorld
        return GSM8KWorld(
            data_path=config.gsm8k_data_path,
            max_examples=config.gsm8k_max_examples,
        )
    return GovernanceWorld(
        crisis_axes=config.world.crisis_axes,
        policy_dims=config.world.policy_dims,
        actions_per_crisis=config.world.actions_per_crisis,
        class_ids=config.committee.class_ids,
    )


def _modify_config(
    config: ExperimentConfig,
    rate: float,
    adv_type: str,
) -> ExperimentConfig:
    """Return a copy with modified corruption parameters."""
    c = copy.deepcopy(config)
    c.corruption.corruption_rate = rate
    c.corruption.adversary_type = adv_type
    return c


def _recording_filename(rate: float, adv_type: str, run_idx: int) -> str:
    """Deterministic recording filename for a sweep condition."""
    return f"rate_{rate:.2f}_adv_{adv_type}_run_{run_idx}.jsonl"


def _compute_summary(
    agg_all: pd.DataFrame,
    ci_alpha: float = 0.1,
) -> pd.DataFrame:
    """Group by condition and compute robust summary stats.

    ci_alpha: bootstrap CI significance level (0.1 = 90% CI, 0.05 = 95% CI).
    Stored in output so downstream analysis can derive z-value.
    """
    groups = agg_all.groupby(["corruption_rate", "adversary_type", "aggregator"])
    rows: List[Dict[str, Any]] = []
    for (rate, adv, agg_name), grp in groups:
        utils = grp["city_utility"].values
        fairs = grp["fairness_jain"].values
        wg = grp["worst_group_utility"].values
        regs = grp["regret"].values

        u_ci = bootstrap_ci(utils, trimmed_mean, alpha=ci_alpha)
        f_ci = bootstrap_ci(fairs, trimmed_mean, alpha=ci_alpha)

        rows.append({
            "corruption_rate": rate,
            "adversary_type": adv,
            "aggregator": agg_name,
            "mean_utility": float(utils.mean()),
            "trimmed_mean_utility": trimmed_mean(utils),
            "ci_low_utility": u_ci[0],
            "ci_high_utility": u_ci[1],
            "mean_fairness": float(fairs.mean()),
            "trimmed_mean_fairness": trimmed_mean(fairs),
            "ci_low_fairness": f_ci[0],
            "ci_high_fairness": f_ci[1],
            "mean_worst_group": float(wg.mean()),
            "mean_regret": float(regs.mean()),
            "ci_alpha": ci_alpha,
        })
    return pd.DataFrame(rows)


async def run_corruption_sweep(
    config: ExperimentConfig,
    mode: str = "record",
    replay_dir: Optional[str] = None,
) -> None:
    """Main sweep: corruption_rate x adversary_type x aggregator x runs."""
    world = _create_world(config)
    out_dir = ensure_dir(config.output_dir)
    rec_dir = os.path.join(out_dir, "recordings")
    os.makedirs(rec_dir, exist_ok=True)

    all_agg: List[pd.DataFrame] = []
    all_agent: List[pd.DataFrame] = []

    total_runs = (
        len(config.corruption_rates)
        * len(config.adversary_types)
        * config.num_runs
    )
    run_count = 0

    for rate in config.corruption_rates:
        for adv_type in config.adversary_types:
            for run_idx in range(config.num_runs):
                seed = derive_seed(config.seed, rate, adv_type, run_idx)
                run_config = _modify_config(config, rate, adv_type)
                run_config.seed = seed
                rec_file = _recording_filename(rate, adv_type, run_idx)

                run_count += 1
                print(
                    f"[SWEEP {run_count}/{total_runs}] "
                    f"rate={rate:.2f}, adv={adv_type}, run={run_idx}"
                )

                if mode == "replay":
                    source_dir = replay_dir or rec_dir
                    rec_path = os.path.join(source_dir, rec_file)
                    if not os.path.exists(rec_path):
                        print(f"  SKIP: recording not found at {rec_path}")
                        continue
                    result = await replay_simulation(run_config, world, rec_path)
                else:
                    rec_path = os.path.join(rec_dir, rec_file)
                    result = await run_hierarchical_simulation(
                        run_config, world, seed, recording_path=rec_path,
                    )

                agg = result.aggregator_log.copy()
                agg["corruption_rate"] = rate
                agg["adversary_type"] = adv_type
                agg["run"] = run_idx
                all_agg.append(agg)

                agent = result.agent_log.copy()
                agent["corruption_rate"] = rate
                agent["adversary_type"] = adv_type
                agent["run"] = run_idx
                all_agent.append(agent)

    # Concatenate
    agg_all = pd.concat(all_agg, ignore_index=True)
    agent_all = pd.concat(all_agent, ignore_index=True)

    # Summary
    summary = _compute_summary(agg_all)

    # Save
    agg_all.to_csv(os.path.join(out_dir, "sweep_aggregator_log.csv"), index=False)
    agent_all.to_csv(os.path.join(out_dir, "sweep_agent_log.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)
    save_config(config, os.path.join(out_dir, "config_used.yaml"))

    # Plot
    from ..plotting.corruption_sweep import plot_corruption_sweep
    plot_corruption_sweep(summary, str(out_dir))

    # Regime analysis
    from ..metrics.regime_analysis import (
        analyze_regimes, format_regime_report, format_regime_map_markdown,
    )
    report = analyze_regimes(summary)
    print(format_regime_report(report))
    report.regime_map.to_csv(os.path.join(out_dir, "regime_map.csv"))
    report.regime_map_detailed.to_csv(
        os.path.join(out_dir, "regime_map_detailed.csv"), index=False)
    with open(os.path.join(out_dir, "regime_map.md"), "w") as f:
        f.write(format_regime_map_markdown(report))

    # Winner bands plot
    from ..plotting.regime_plots import plot_winner_bands
    plot_winner_bands(summary, str(out_dir))

    print(f"\nResults saved to {out_dir}/")
