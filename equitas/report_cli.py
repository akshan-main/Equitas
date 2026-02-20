"""CLI for generating a JSON benchmark report: python -m equitas report --results-dir <dir>

Outputs equitas_report.json with:
  - benchmark version
  - config digest
  - per-aggregator metric AUCs (area under corruption curve)
  - fairness guardrail flags
  - regime transitions
  - collapse points
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="equitas report",
        description="Generate a JSON benchmark report from sweep results",
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing sweep_summary.csv (and optionally config_used.yaml)",
    )
    parser.add_argument(
        "--metric", type=str, default="trimmed_mean_utility",
        help="Primary metric (default: trimmed_mean_utility)",
    )
    parser.add_argument(
        "--collapse-threshold", type=float, default=0.5,
        help="Collapse detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <results-dir>/equitas_report.json)",
    )

    args = parser.parse_args(sys.argv[2:])
    _generate_report(args)


def _auc_corruption_curve(
    agg_data: pd.DataFrame,
    metric: str,
) -> float:
    """Trapezoidal AUC of metric vs corruption_rate for one aggregator."""
    sorted_data = agg_data.sort_values("corruption_rate")
    rates = sorted_data["corruption_rate"].values
    vals = sorted_data[metric].values
    if len(rates) < 2:
        return float(vals[0]) if len(vals) == 1 else 0.0
    return float(np.trapz(vals, rates))


def _generate_report(args: argparse.Namespace) -> None:
    summary_path = os.path.join(args.results_dir, "sweep_summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)

    summary = pd.read_csv(summary_path)

    # --- Regime analysis ---
    from .metrics.regime_analysis import analyze_regimes
    report = analyze_regimes(
        summary,
        metric=args.metric,
        collapse_threshold=args.collapse_threshold,
    )

    # --- Per-aggregator AUC ---
    auc_entries: List[Dict[str, Any]] = []
    non_oracle = summary[summary["aggregator"] != "oracle_upper_bound"]
    for adv_type in sorted(non_oracle["adversary_type"].unique()):
        adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
        for agg_name in sorted(adv_data["aggregator"].unique()):
            agg_data = adv_data[adv_data["aggregator"] == agg_name]
            auc_entries.append({
                "adversary_type": adv_type,
                "aggregator": agg_name,
                "auc_utility": _auc_corruption_curve(agg_data, args.metric),
                "auc_fairness": _auc_corruption_curve(
                    agg_data, "trimmed_mean_fairness"
                ) if "trimmed_mean_fairness" in agg_data.columns else None,
                "auc_worst_group": _auc_corruption_curve(
                    agg_data, "mean_worst_group"
                ) if "mean_worst_group" in agg_data.columns else None,
            })

    # --- Fairness guardrails ---
    guardrails: List[Dict[str, Any]] = []
    if "mean_worst_group" in non_oracle.columns:
        for adv_type in sorted(non_oracle["adversary_type"].unique()):
            adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
            for agg_name in sorted(adv_data["aggregator"].unique()):
                agg_data = adv_data[adv_data["aggregator"] == agg_name]
                min_wg = float(agg_data["mean_worst_group"].min())
                guardrails.append({
                    "adversary_type": adv_type,
                    "aggregator": agg_name,
                    "min_worst_group_utility": min_wg,
                    "passes_guardrail": min_wg > 0.1,
                })

    # --- Config digest ---
    config_info: Dict[str, Any] = {}
    config_path = os.path.join(args.results_dir, "config_used.yaml")
    if os.path.exists(config_path):
        config_info["config_path"] = config_path
        # Extract key parameters if YAML is available
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            config_info["seed"] = cfg.get("seed")
            config_info["num_runs"] = cfg.get("num_runs")
            config_info["rounds"] = cfg.get("rounds")
        except Exception:
            pass

    # --- Assemble JSON ---
    output: Dict[str, Any] = {
        "equitas_version": __version__,
        "metric": args.metric,
        "collapse_threshold": args.collapse_threshold,
        "ci_alpha": report.ci_alpha,
        "ci_level": report.ci_level,
        "z_value": report.z_value,
        "config": config_info,
        "corruption_rates": sorted(
            float(r) for r in summary["corruption_rate"].unique()
        ),
        "adversary_types": sorted(summary["adversary_type"].unique().tolist()),
        "aggregators": sorted(non_oracle["aggregator"].unique().tolist()),
        "auc_scores": auc_entries,
        "fairness_guardrails": guardrails,
        "regime_transitions": [
            {
                "adversary_type": t.adversary_type,
                "crossover_rate": t.crossover_rate,
                "leader_before": t.leader_before,
                "leader_after": t.leader_after,
                "delta": t.delta,
                "significant": t.significant,
            }
            for t in report.transitions
        ],
        "collapse_points": [
            {
                "adversary_type": c.adversary_type,
                "aggregator": c.aggregator,
                "collapse_rate": c.collapse_rate,
                "rel_perf": c.rel_perf,
                "threshold": c.threshold,
            }
            for c in report.collapses
        ],
        "regime_map_welfare": {},
    }

    # Regime map as nested dict
    if not report.regime_map.empty:
        for adv_type in report.regime_map.index:
            row = report.regime_map.loc[adv_type]
            output["regime_map_welfare"][adv_type] = {
                col: row.get(col, "") for col in ["low", "mid", "high"]
            }

    # Write
    out_path = args.output or os.path.join(
        args.results_dir, "equitas_report.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Report saved to {out_path}")
