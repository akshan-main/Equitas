"""
Full-Hierarchy vs flat architecture comparison.
Same as hierarchical_vs_flat.py but the hierarchical arm uses engine_fh
where each aggregator controls both intra-class and inter-class aggregation.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ExperimentConfig, save_config
from ..simulation.engine_fh import run_fh_simulation
from ..simulation.flat_engine import run_flat_simulation
from ..simulation.replay_fh import replay_fh_simulation
from ..utils import derive_seed, ensure_dir
from ..worlds.governance import GovernanceWorld


async def run_fh_hierarchical_vs_flat(
    config: ExperimentConfig,
    mode: str = "record",
    replay_dir: Optional[str] = None,
) -> None:
    """
    Run same corruption conditions with FH-hierarchical and flat architectures.
    FH-hierarchical: each aggregator controls both levels independently.
    Flat: all agents vote directly (no hierarchy).
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
    total = len(config.corruption_rates) * config.num_runs * 2
    count = 0

    for rate in config.corruption_rates:
        for run_idx in range(config.num_runs):
            seed = derive_seed(config.seed, "hvf_fh", rate, run_idx)
            c = copy.deepcopy(config)
            c.corruption.corruption_rate = rate

            rec_file_h = f"hvf_fh_hier_rate{rate:.2f}_run{run_idx}.jsonl"

            # Full-Hierarchy
            count += 1
            print(f"[FH-HIER_VS_FLAT {count}/{total}] rate={rate:.2f}, run={run_idx}, arch=fh_hierarchical")

            if mode == "replay":
                source_dir = replay_dir or rec_dir
                rec_path = os.path.join(source_dir, rec_file_h)
                if not os.path.exists(rec_path):
                    print(f"  SKIP: recording not found at {rec_path}")
                    h_result = None
                else:
                    h_result = await replay_fh_simulation(c, world, rec_path)
            else:
                rec_path = os.path.join(rec_dir, rec_file_h)
                h_result = await run_fh_simulation(c, world, seed, recording_path=rec_path)

            if h_result is not None:
                for agg_name in h_result.aggregator_log["aggregator"].unique():
                    sub = h_result.aggregator_log[h_result.aggregator_log["aggregator"] == agg_name]
                    rows.append({
                        "corruption_rate": rate,
                        "run": run_idx,
                        "architecture": "fh_hierarchical",
                        "aggregator": agg_name,
                        "mean_utility": float(sub["city_utility"].mean()),
                        "mean_fairness": float(sub["fairness_jain"].mean()),
                        "mean_worst_group": float(sub["worst_group_utility"].mean()),
                    })

            # Flat (no replay support — flat engine doesn't produce recordings)
            count += 1
            print(f"[FH-HIER_VS_FLAT {count}/{total}] rate={rate:.2f}, run={run_idx}, arch=flat")
            f_result = await run_flat_simulation(c, world, seed)
            for agg_name in f_result.aggregator_log["aggregator"].unique():
                sub = f_result.aggregator_log[f_result.aggregator_log["aggregator"] == agg_name]
                rows.append({
                    "corruption_rate": rate,
                    "run": run_idx,
                    "architecture": "flat",
                    "aggregator": agg_name,
                    "mean_utility": float(sub["city_utility"].mean()),
                    "mean_fairness": float(sub["fairness_jain"].mean()),
                    "mean_worst_group": float(sub["worst_group_utility"].mean()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "fh_hierarchical_vs_flat.csv"), index=False)
    save_config(config, os.path.join(out_dir, "config_used.yaml"))

    print(f"\nFH Hierarchical vs Flat results saved to {out_dir}/")
