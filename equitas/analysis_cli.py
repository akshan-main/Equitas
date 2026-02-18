"""CLI for post-hoc analysis: python -m equitas analyze regime-map --results-dir <dir>"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="equitas analyze",
        description="Post-hoc analysis of Equitas experiment results",
    )
    sub = parser.add_subparsers(dest="command")

    # regime-map subcommand
    regime = sub.add_parser("regime-map", help="Generate regime map from sweep results")
    regime.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing sweep_summary.csv",
    )
    regime.add_argument(
        "--metric", type=str, default="trimmed_mean_utility",
        help="Primary metric for analysis (default: trimmed_mean_utility)",
    )
    regime.add_argument(
        "--collapse-threshold", type=float, default=0.5,
        help="Relative performance threshold for collapse detection (default: 0.5)",
    )

    # Drop the 'analyze' arg that __main__.py already consumed
    import sys
    args = parser.parse_args(sys.argv[2:])

    if args.command == "regime-map":
        _run_regime_map(args)
    else:
        parser.print_help()


def _run_regime_map(args: argparse.Namespace) -> None:
    from .metrics.regime_analysis import (
        analyze_regimes,
        format_regime_report,
        format_regime_map_markdown,
    )
    from .plotting.regime_plots import plot_winner_bands

    summary_path = os.path.join(args.results_dir, "sweep_summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found")
        return

    summary = pd.read_csv(summary_path)
    report = analyze_regimes(
        summary,
        metric=args.metric,
        collapse_threshold=args.collapse_threshold,
    )

    # Print to console
    print(format_regime_report(report))

    # Save CSV
    out = args.results_dir
    report.regime_map.to_csv(os.path.join(out, "regime_map.csv"))
    report.regime_map_detailed.to_csv(
        os.path.join(out, "regime_map_detailed.csv"), index=False)

    # Save markdown
    md_path = os.path.join(out, "regime_map.md")
    with open(md_path, "w") as f:
        f.write(format_regime_map_markdown(report))
    print(f"\nSaved {md_path}")

    # Plot
    plot_winner_bands(summary, out, metric=args.metric)

    print(f"\nAll regime analysis artifacts saved to {out}/")
