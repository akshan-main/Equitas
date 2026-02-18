"""Top-level experiment runner. Reads config and dispatches."""
from __future__ import annotations

import argparse
import asyncio

from ..config import load_config, validate_config
from .hierarchical_vs_flat import run_hierarchical_vs_flat
from .hierarchical_vs_flat_fh import run_fh_hierarchical_vs_flat
from .pareto_sweep import run_pareto_sweep
from .pareto_sweep_fh import run_fh_pareto_sweep
from .recovery import run_recovery_experiment
from .recovery_fh import run_fh_recovery_experiment
from .scaling import run_scaling_experiment
from .scaling_fh import run_fh_scaling_experiment
from .sweep import run_corruption_sweep
from .sweep_fh import run_fh_corruption_sweep

DISPATCH = {
    "sweep": run_corruption_sweep,
    "sweep_fh": run_fh_corruption_sweep,
    "pareto": run_pareto_sweep,
    "pareto_fh": run_fh_pareto_sweep,
    "recovery": run_recovery_experiment,
    "recovery_fh": run_fh_recovery_experiment,
    "scaling": run_scaling_experiment,
    "scaling_fh": run_fh_scaling_experiment,
    "hierarchical_vs_flat": run_hierarchical_vs_flat,
    "hierarchical_vs_flat_fh": run_fh_hierarchical_vs_flat,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Equitas benchmark runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode", type=str, default="record",
        choices=["record", "replay"],
        help="record = run LLMs + save recordings; replay = re-run aggregation from saved recordings (zero API calls)",
    )
    parser.add_argument(
        "--replay-dir", type=str, default=None,
        help="Directory containing JSONL recordings from a previous record run. Required for --mode replay.",
    )
    args = parser.parse_args()

    if args.mode == "replay" and args.replay_dir is None:
        parser.error("--replay-dir is required when --mode is 'replay'")

    config = load_config(args.config)
    validate_config(config)

    runner = DISPATCH.get(config.experiment_type)
    if runner is None:
        raise ValueError(
            f"Unknown experiment_type: {config.experiment_type}. "
            f"Valid: {list(DISPATCH.keys())}"
        )

    mode_label = "RECORD" if args.mode == "record" else "REPLAY (zero API calls)"
    print(f"=== Equitas: {config.name} ({config.experiment_type}) [{mode_label}] ===")
    asyncio.run(runner(config, mode=args.mode, replay_dir=args.replay_dir))
    print("=== Done ===")
