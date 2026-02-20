"""Alpha/beta Pareto sweep for welfare-fairness tradeoff."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import AggregatorConfig, ExperimentConfig, save_config
from ..simulation.engine import run_hierarchical_simulation
from ..simulation.replay import replay_simulation
from ..utils import derive_seed, ensure_dir
from ..worlds.governance import GovernanceWorld


async def run_pareto_sweep(
    config: ExperimentConfig,
    mode: str = "record",
    replay_dir: Optional[str] = None,
) -> None:
    """
    For each (alpha, beta), run simulation with MW using that loss combo.
    Record resulting welfare and fairness.
    """
    world = GovernanceWorld(
        crisis_axes=config.world.crisis_axes,
        policy_dims=config.world.policy_dims,
        actions_per_crisis=config.world.actions_per_crisis,
        class_ids=config.committee.class_ids,
    )
    out_dir = ensure_dir(config.output_dir)
    rec_dir = os.path.join(out_dir, "recordings")
    os.makedirs(rec_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(config.alpha_values) * len(config.beta_values) * config.num_runs
    count = 0

    for alpha in config.alpha_values:
        for beta in config.beta_values:
            for run_idx in range(config.num_runs):
                count += 1
                seed = derive_seed(config.seed, alpha, beta, run_idx)

                # Modify config: only run MW with this alpha/beta
                c = copy.deepcopy(config)
                c.aggregators = [
                    AggregatorConfig(
                        method="multiplicative_weights",
                        eta=1.0, alpha=alpha, beta=beta,
                    ),
                    AggregatorConfig(method="majority_vote"),
                ]

                rec_file = f"pareto_a{alpha:.2f}_b{beta:.2f}_run{run_idx}.jsonl"
                print(f"[PARETO {count}/{total}] alpha={alpha:.2f}, beta={beta:.2f}, run={run_idx}")

                if mode == "replay":
                    source_dir = replay_dir or rec_dir
                    rec_path = os.path.join(source_dir, rec_file)
                    if not os.path.exists(rec_path):
                        print(f"  SKIP: recording not found at {rec_path}")
                        continue
                    result = await replay_simulation(c, world, rec_path)
                else:
                    rec_path = os.path.join(rec_dir, rec_file)
                    result = await run_hierarchical_simulation(c, world, seed, recording_path=rec_path)

                agg = result.aggregator_log

                for agg_name in agg["aggregator"].unique():
                    sub = agg[agg["aggregator"] == agg_name]
                    rows.append({
                        "alpha": alpha,
                        "beta": beta,
                        "run": run_idx,
                        "aggregator": agg_name,
                        "mean_utility": float(sub["city_utility"].mean()),
                        "mean_fairness": float(sub["fairness_jain"].mean()),
                        "mean_worst_group": float(sub["worst_group_utility"].mean()),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "pareto_results.csv"), index=False)
    save_config(config, os.path.join(out_dir, "config_used.yaml"))

    from ..plotting.pareto_plots import plot_pareto_frontier
    plot_pareto_frontier(df, str(out_dir))

    print(f"\nPareto results saved to {out_dir}/")
