"""
Full-Hierarchy mid-run corruption onset experiment.
Same as recovery.py but uses engine_fh where each aggregator
controls both intra-class and inter-class aggregation.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, save_config
from ..simulation.engine_fh import run_fh_simulation
from ..simulation.replay_fh import replay_fh_simulation
from ..utils import derive_seed, ensure_dir
from ..worlds.governance import GovernanceWorld


async def run_fh_recovery_experiment(
    config: ExperimentConfig,
    mode: str = "record",
    replay_dir: Optional[str] = None,
) -> None:
    """
    Full-Hierarchy recovery: corruption activates at round T/2.
    Track utility per round and MW weight evolution.
    Each aggregator independently controls both levels.
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

    # Set corruption onset to midpoint
    onset = config.corruption.corruption_onset_round
    if onset is None:
        onset = config.world.num_rounds // 2
    c = copy.deepcopy(config)
    c.corruption.corruption_onset_round = onset

    all_agg: List[pd.DataFrame] = []
    all_weights: List[Dict[str, Any]] = []

    for run_idx in range(config.num_runs):
        seed = derive_seed(config.seed, "recovery_fh", run_idx)
        rec_file = f"recovery_fh_run_{run_idx}.jsonl"
        print(f"[FH-RECOVERY run={run_idx}] onset_round={onset}")

        if mode == "replay":
            source_dir = replay_dir or rec_dir
            rec_path = os.path.join(source_dir, rec_file)
            if not os.path.exists(rec_path):
                print(f"  SKIP: recording not found at {rec_path}")
                continue
            result = await replay_fh_simulation(c, world, rec_path)
        else:
            rec_path = os.path.join(rec_dir, rec_file)
            result = await run_fh_simulation(c, world, seed, recording_path=rec_path)

        agg = result.aggregator_log.copy()
        agg["run"] = run_idx
        agg["onset_round"] = onset
        all_agg.append(agg)

        # Weight history
        for agg_name, weight_list in result.weight_history.items():
            for round_id, weights in enumerate(weight_list):
                for agent_idx, w in enumerate(weights):
                    all_weights.append({
                        "run": run_idx,
                        "round_id": round_id,
                        "aggregator": agg_name,
                        "agent_index": agent_idx,
                        "weight": float(w),
                        "onset_round": onset,
                    })

    agg_all = pd.concat(all_agg, ignore_index=True)
    weights_df = pd.DataFrame(all_weights)

    agg_all.to_csv(os.path.join(out_dir, "recovery_fh_aggregator_log.csv"), index=False)
    weights_df.to_csv(os.path.join(out_dir, "recovery_fh_weight_history.csv"), index=False)
    save_config(config, os.path.join(out_dir, "config_used.yaml"))

    from ..plotting.recovery_plots import plot_recovery_trajectory
    plot_recovery_trajectory(agg_all, weights_df, onset, str(out_dir))

    print(f"\nFH Recovery results saved to {out_dir}/")
