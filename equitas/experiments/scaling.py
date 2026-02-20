"""Committee size scaling experiment."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ExperimentConfig, save_config
from ..simulation.engine import run_hierarchical_simulation
from ..simulation.replay import replay_simulation
from ..utils import derive_seed, ensure_dir
from ..worlds.governance import GovernanceWorld


async def run_scaling_experiment(
    config: ExperimentConfig,
    mode: str = "record",
    replay_dir: Optional[str] = None,
) -> None:
    """
    For each N in committee_sizes, set members_per_class = N,
    run simulation, compare robustness across sizes.
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
    total = len(config.committee_sizes) * config.num_runs
    count = 0

    for N in config.committee_sizes:
        for run_idx in range(config.num_runs):
            count += 1
            seed = derive_seed(config.seed, "scaling", N, run_idx)

            c = copy.deepcopy(config)
            c.committee.members_per_class = N

            rec_file = f"scaling_N{N}_run{run_idx}.jsonl"
            print(f"[SCALING {count}/{total}] N={N}, run={run_idx}")

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
                    "members_per_class": N,
                    "run": run_idx,
                    "aggregator": agg_name,
                    "mean_utility": float(sub["city_utility"].mean()),
                    "mean_fairness": float(sub["fairness_jain"].mean()),
                    "mean_worst_group": float(sub["worst_group_utility"].mean()),
                    "mean_regret": float(sub["regret"].mean()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "scaling_results.csv"), index=False)
    save_config(config, os.path.join(out_dir, "config_used.yaml"))

    from ..plotting.scaling_plots import plot_scaling
    plot_scaling(df, str(out_dir))

    print(f"\nScaling results saved to {out_dir}/")
