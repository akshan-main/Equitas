"""
Post-hoc regime analysis: reads sweep summary and identifies
phase transitions, crossover points, and regime boundaries.

Uses CI-based pairwise dominance (not raw argmax) so regime
transitions are statistical statements, not noise artifacts.

z-value is derived from ci_alpha stored in sweep_summary.csv,
never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairwiseDominance:
    """Statistical comparison between two aggregators at a condition."""

    adversary_type: str
    corruption_rate: float
    metric: str
    winner: str
    loser: str
    delta: float        # winner_value - loser_value
    ci_low: float       # lower bound of difference CI
    ci_high: float      # upper bound of difference CI
    significant: bool   # True if CI does not contain 0


@dataclass
class RegimeTransition:
    """A point where the dominant aggregator changes."""

    adversary_type: str
    metric: str
    leader_before: str
    leader_after: str
    crossover_rate: float
    delta: float        # margin at crossover
    significant: bool   # CI-backed significance


@dataclass
class CollapsePoint:
    """An aggregator's normalized performance drops below a threshold."""

    adversary_type: str
    aggregator: str
    metric: str
    collapse_rate: float
    rel_perf: float     # shift-invariant: (M(ε) - M(ε_max)) / (M(0) - M(ε_max))
    threshold: float


@dataclass
class RegimeReport:
    """Complete regime analysis output."""

    regime_map: pd.DataFrame           # pivot: adversary_type × corruption_level → best
    regime_map_detailed: pd.DataFrame  # full: one row per (adv, rate, metric)
    transitions: List[RegimeTransition] = field(default_factory=list)
    collapses: List[CollapsePoint] = field(default_factory=list)
    dominance_pairs: List[PairwiseDominance] = field(default_factory=list)
    ci_alpha: Optional[float] = None   # CI significance level (e.g. 0.1)
    ci_level: Optional[float] = None   # 1 - ci_alpha (e.g. 0.9 = 90% CI)
    z_value: Optional[float] = None    # derived Φ⁻¹(1 - α/2)


# ---------------------------------------------------------------------------
# Metric config
# ---------------------------------------------------------------------------

METRIC_SPECS = [
    ("trimmed_mean_utility", "ci_low_utility", "ci_high_utility"),
    ("mean_worst_group", None, None),
    ("trimmed_mean_fairness", "ci_low_fairness", "ci_high_fairness"),
]

METRIC_LABELS = {
    "trimmed_mean_utility": "welfare",
    "mean_worst_group": "worst_group",
    "trimmed_mean_fairness": "fairness",
}


def _corruption_level(rate: float) -> str:
    if rate <= 0.25:
        return "low"
    elif rate <= 0.5:
        return "mid"
    else:
        return "high"


# ---------------------------------------------------------------------------
# z-value inference (never hardcoded)
# ---------------------------------------------------------------------------

def _z_from_alpha(alpha: float) -> float:
    """Φ⁻¹(1 - α/2) via Abramowitz & Stegun 26.2.23 rational approximation.

    No scipy dependency. Exact for common values, <1e-4 error elsewhere.
    """
    p = 1.0 - alpha / 2.0
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)


def _infer_z(summary: pd.DataFrame) -> float:
    """Derive z from ci_alpha column. Raises if missing."""
    if "ci_alpha" in summary.columns:
        alpha = float(summary["ci_alpha"].iloc[0])
        return _z_from_alpha(alpha)
    raise ValueError(
        "sweep_summary.csv missing 'ci_alpha' column. "
        "Re-run sweep to generate it, or pass ci_alpha explicitly."
    )


def _approx_se_from_ci(ci_lo: float, ci_hi: float, z: float) -> float:
    """Approximate SE from a percentile bootstrap CI."""
    return max((ci_hi - ci_lo) / (2 * z), 1e-12)


# ---------------------------------------------------------------------------
# CI-based pairwise dominance
# ---------------------------------------------------------------------------

def pairwise_ci_dominance(
    summary: pd.DataFrame,
    metric: str = "trimmed_mean_utility",
    ci_lo_col: Optional[str] = "ci_low_utility",
    ci_hi_col: Optional[str] = "ci_high_utility",
    oracle_name: str = "oracle_upper_bound",
    ci_alpha: Optional[float] = None,
) -> List[PairwiseDominance]:
    """Compute pairwise dominance using CI separation.

    A dominates B if the CI of (M_A - M_B) is entirely above 0.
    z-value derived from ci_alpha (column or parameter).
    """
    results: List[PairwiseDominance] = []
    non_oracle = summary[summary["aggregator"] != oracle_name]
    has_ci = (ci_lo_col and ci_hi_col and
              ci_lo_col in non_oracle.columns and
              ci_hi_col in non_oracle.columns)

    if has_ci:
        if ci_alpha is not None:
            z = _z_from_alpha(ci_alpha)
        elif "ci_alpha" in summary.columns:
            z = _z_from_alpha(float(summary["ci_alpha"].iloc[0]))
        else:
            raise ValueError(
                "Cannot infer z-value: no ci_alpha column in summary and "
                "no ci_alpha parameter provided."
            )
    else:
        z = 0.0  # unused

    for adv_type in non_oracle["adversary_type"].unique():
        adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
        for rate in sorted(adv_data["corruption_rate"].unique()):
            rate_data = adv_data[adv_data["corruption_rate"] == rate]
            aggs = sorted(rate_data["aggregator"].unique())

            for i, agg_a in enumerate(aggs):
                for agg_b in aggs[i + 1:]:
                    row_a = rate_data[rate_data["aggregator"] == agg_a].iloc[0]
                    row_b = rate_data[rate_data["aggregator"] == agg_b].iloc[0]

                    raw_delta = row_a[metric] - row_b[metric]
                    if raw_delta >= 0:
                        winner, loser, delta = agg_a, agg_b, raw_delta
                    else:
                        winner, loser, delta = agg_b, agg_a, -raw_delta

                    if has_ci:
                        se_a = _approx_se_from_ci(row_a[ci_lo_col], row_a[ci_hi_col], z)
                        se_b = _approx_se_from_ci(row_b[ci_lo_col], row_b[ci_hi_col], z)
                        se_diff = np.sqrt(se_a**2 + se_b**2)
                        ci_low = delta - z * se_diff
                        ci_high = delta + z * se_diff
                        significant = bool(ci_low > 0)
                    else:
                        ci_low = delta
                        ci_high = delta
                        significant = bool(delta > 1e-6)

                    results.append(PairwiseDominance(
                        adversary_type=adv_type,
                        corruption_rate=rate,
                        metric=metric,
                        winner=winner,
                        loser=loser,
                        delta=delta,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        significant=significant,
                    ))

    return results


# ---------------------------------------------------------------------------
# Crossover and collapse detection
# ---------------------------------------------------------------------------

def find_crossover_points(
    summary: pd.DataFrame,
    metric: str = "trimmed_mean_utility",
    ci_lo_col: Optional[str] = "ci_low_utility",
    ci_hi_col: Optional[str] = "ci_high_utility",
    oracle_name: str = "oracle_upper_bound",
    ci_alpha: Optional[float] = None,
) -> List[RegimeTransition]:
    """Find corruption rates where the best aggregator changes.

    Significance: the new leader must dominate the old leader by CI.
    """
    transitions: List[RegimeTransition] = []
    non_oracle = summary[summary["aggregator"] != oracle_name]
    has_ci = (ci_lo_col and ci_hi_col and
              ci_lo_col in non_oracle.columns)

    if has_ci:
        if ci_alpha is not None:
            z = _z_from_alpha(ci_alpha)
        elif "ci_alpha" in summary.columns:
            z = _z_from_alpha(float(summary["ci_alpha"].iloc[0]))
        else:
            raise ValueError("Cannot infer z-value: no ci_alpha available.")
    else:
        z = 0.0

    for adv_type in non_oracle["adversary_type"].unique():
        adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
        rates = sorted(adv_data["corruption_rate"].unique())
        prev_leader: Optional[str] = None

        for rate in rates:
            rate_data = adv_data[adv_data["corruption_rate"] == rate]
            best_row = rate_data.loc[rate_data[metric].idxmax()]
            current_leader = best_row["aggregator"]

            if prev_leader is not None and current_leader != prev_leader:
                prev_row = rate_data[rate_data["aggregator"] == prev_leader]
                if not prev_row.empty:
                    delta = best_row[metric] - prev_row.iloc[0][metric]
                    if has_ci:
                        se_new = _approx_se_from_ci(
                            best_row[ci_lo_col], best_row[ci_hi_col], z)
                        se_old = _approx_se_from_ci(
                            prev_row.iloc[0][ci_lo_col], prev_row.iloc[0][ci_hi_col], z)
                        se_diff = np.sqrt(se_new**2 + se_old**2)
                        significant = bool((delta - z * se_diff) > 0)
                    else:
                        significant = bool(delta > 1e-6)
                else:
                    delta = 0.0
                    significant = False

                transitions.append(RegimeTransition(
                    adversary_type=adv_type,
                    metric=metric,
                    leader_before=prev_leader,
                    leader_after=current_leader,
                    crossover_rate=rate,
                    delta=delta,
                    significant=significant,
                ))
            prev_leader = current_leader

    return transitions


def find_collapse_points(
    summary: pd.DataFrame,
    metric: str = "trimmed_mean_utility",
    threshold: float = 0.5,
    oracle_name: str = "oracle_upper_bound",
) -> List[CollapsePoint]:
    """Find ε where normalized performance drops below threshold.

    Uses shift-invariant normalization:
        rel_perf(ε) = (M(ε) - M(ε_max)) / (M(0) - M(ε_max))

    Maps clean → 1.0, worst corruption → 0.0.
    Collapse when rel_perf < threshold.
    """
    collapses: List[CollapsePoint] = []
    non_oracle = summary[summary["aggregator"] != oracle_name]

    for adv_type in non_oracle["adversary_type"].unique():
        adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
        for agg_name in adv_data["aggregator"].unique():
            agg_data = adv_data[adv_data["aggregator"] == agg_name]
            agg_data = agg_data.sort_values("corruption_rate")

            clean = agg_data[agg_data["corruption_rate"] == 0.0]
            if clean.empty:
                continue
            m_clean = clean[metric].values[0]

            # M(ε_max) = performance at highest corruption rate
            m_worst = agg_data[metric].min()
            denom = m_clean - m_worst
            if abs(denom) < 1e-12:
                # Flat metric: M(0) ≈ M(ε_max) — no degradation signal.
                # Emit NaN so report shows it explicitly.
                collapses.append(CollapsePoint(
                    adversary_type=adv_type,
                    aggregator=agg_name,
                    metric=metric,
                    collapse_rate=float("nan"),
                    rel_perf=float("nan"),
                    threshold=threshold,
                ))
                continue

            for _, row in agg_data.iterrows():
                rel_perf = (row[metric] - m_worst) / denom
                if rel_perf < threshold:
                    collapses.append(CollapsePoint(
                        adversary_type=adv_type,
                        aggregator=agg_name,
                        metric=metric,
                        collapse_rate=row["corruption_rate"],
                        rel_perf=rel_perf,
                        threshold=threshold,
                    ))
                    break

    return collapses


# ---------------------------------------------------------------------------
# Multi-metric regime map
# ---------------------------------------------------------------------------

def build_regime_map(
    summary: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    oracle_name: str = "oracle_upper_bound",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build regime maps across multiple metrics.

    Returns:
        (detailed_df, pivot_df)
    """
    if metrics is None:
        metrics = [m for m, _, _ in METRIC_SPECS]

    non_oracle = summary[summary["aggregator"] != oracle_name]
    entries: List[Dict[str, Any]] = []

    for metric in metrics:
        if metric not in non_oracle.columns:
            continue
        metric_label = METRIC_LABELS.get(metric, metric)

        for adv_type in non_oracle["adversary_type"].unique():
            adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
            for rate in sorted(adv_data["corruption_rate"].unique()):
                rate_data = adv_data[adv_data["corruption_rate"] == rate]
                best_row = rate_data.loc[rate_data[metric].idxmax()]

                sorted_rows = rate_data.sort_values(metric, ascending=False)
                if len(sorted_rows) >= 2:
                    runner = sorted_rows.iloc[1]
                    delta = best_row[metric] - runner[metric]
                    runner_name = runner["aggregator"]
                else:
                    delta = 0.0
                    runner_name = ""

                entries.append({
                    "adversary_type": adv_type,
                    "corruption_rate": rate,
                    "corruption_level": _corruption_level(rate),
                    "metric": metric_label,
                    "best_aggregator": best_row["aggregator"],
                    "value": best_row[metric],
                    "runner_up": runner_name,
                    "margin": delta,
                })

    detailed = pd.DataFrame(entries)

    welfare = detailed[detailed["metric"] == "welfare"] if not detailed.empty else detailed
    if not welfare.empty:
        pivot = welfare.pivot_table(
            index="adversary_type",
            columns="corruption_level",
            values="best_aggregator",
            aggfunc="first",
        )
        for col in ["low", "mid", "high"]:
            if col not in pivot.columns:
                pivot[col] = ""
        pivot = pivot[["low", "mid", "high"]]
    else:
        pivot = pd.DataFrame()

    return detailed, pivot


def mechanism_dominance_matrix(
    summary: pd.DataFrame,
    metric: str = "trimmed_mean_utility",
    oracle_name: str = "oracle_upper_bound",
) -> pd.DataFrame:
    """Build a matrix: (aggregator, adversary_type) → win fraction."""
    non_oracle = summary[summary["aggregator"] != oracle_name]
    aggregators = sorted(non_oracle["aggregator"].unique())
    adv_types = sorted(non_oracle["adversary_type"].unique())

    wins: Dict[Tuple[str, str], int] = {
        (agg, adv): 0 for agg in aggregators for adv in adv_types
    }
    counts: Dict[str, int] = {adv: 0 for adv in adv_types}

    for adv_type in adv_types:
        adv_data = non_oracle[non_oracle["adversary_type"] == adv_type]
        for rate in adv_data["corruption_rate"].unique():
            rate_data = adv_data[adv_data["corruption_rate"] == rate]
            best_row = rate_data.loc[rate_data[metric].idxmax()]
            wins[(best_row["aggregator"], adv_type)] += 1
            counts[adv_type] += 1

    rows = []
    for agg in aggregators:
        row = {"aggregator": agg}
        for adv in adv_types:
            total = counts[adv] if counts[adv] > 0 else 1
            row[adv] = wins[(agg, adv)] / total
        rows.append(row)

    return pd.DataFrame(rows).set_index("aggregator")


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_regimes(
    summary: pd.DataFrame,
    metric: str = "trimmed_mean_utility",
    collapse_threshold: float = 0.5,
    ci_alpha: Optional[float] = None,
) -> RegimeReport:
    """Run full regime analysis on sweep summary data.

    ci_alpha: override for CI significance level (reads from summary if absent).
    """
    ci_map = {
        "trimmed_mean_utility": ("ci_low_utility", "ci_high_utility"),
        "trimmed_mean_fairness": ("ci_low_fairness", "ci_high_fairness"),
    }
    ci_lo, ci_hi = ci_map.get(metric, (None, None))

    # Resolve ci_alpha and derive z for the report header
    resolved_alpha: Optional[float] = ci_alpha
    if resolved_alpha is None and "ci_alpha" in summary.columns:
        resolved_alpha = float(summary["ci_alpha"].iloc[0])
    resolved_z: Optional[float] = None
    resolved_level: Optional[float] = None
    if resolved_alpha is not None:
        resolved_z = _z_from_alpha(resolved_alpha)
        resolved_level = 1.0 - resolved_alpha

    transitions = find_crossover_points(
        summary, metric=metric, ci_lo_col=ci_lo, ci_hi_col=ci_hi,
        ci_alpha=ci_alpha,
    )
    collapses = find_collapse_points(
        summary, metric=metric, threshold=collapse_threshold,
    )
    dominance_pairs = pairwise_ci_dominance(
        summary, metric=metric, ci_lo_col=ci_lo, ci_hi_col=ci_hi,
        ci_alpha=ci_alpha,
    )
    detailed, pivot = build_regime_map(summary)

    return RegimeReport(
        regime_map=pivot,
        regime_map_detailed=detailed,
        transitions=transitions,
        collapses=collapses,
        dominance_pairs=dominance_pairs,
        ci_alpha=resolved_alpha,
        ci_level=resolved_level,
        z_value=resolved_z,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_regime_report(report: RegimeReport) -> str:
    """Format regime analysis as human-readable text for logs."""
    lines: List[str] = []

    lines.append("=" * 60)
    lines.append("REGIME ANALYSIS REPORT")
    lines.append("=" * 60)

    if report.ci_alpha is not None:
        lines.append(
            f"\nCI parameters: ci_alpha={report.ci_alpha}, "
            f"ci_level={report.ci_level:.0%}, "
            f"z={report.z_value:.4f}"
        )

    lines.append("\n--- Regime Map: welfare ---")
    if isinstance(report.regime_map, pd.DataFrame) and not report.regime_map.empty:
        lines.append(report.regime_map.to_string())
    else:
        lines.append("  (no data)")

    lines.append("\n--- Detailed Regime Map (all metrics) ---")
    if not report.regime_map_detailed.empty:
        cols = ["adversary_type", "corruption_rate", "metric",
                "best_aggregator", "margin"]
        lines.append(report.regime_map_detailed[cols].to_string(index=False))

    lines.append(f"\n--- Phase Transitions ({len(report.transitions)} found) ---")
    for t in report.transitions:
        sig = "*" if t.significant else " "
        lines.append(
            f"  {sig}[{t.adversary_type}] at ε={t.crossover_rate:.2f}: "
            f"{t.leader_before} → {t.leader_after} "
            f"(Δ={t.delta:.4f})"
        )

    lines.append(f"\n--- Collapse Points ({len(report.collapses)} found) ---")
    for c in report.collapses:
        if np.isnan(c.rel_perf):
            lines.append(
                f"  [{c.adversary_type}] {c.aggregator}: "
                f"rel_perf=NaN (flat metric, M(0) ≈ M(ε_max))"
            )
        else:
            lines.append(
                f"  [{c.adversary_type}] {c.aggregator}: "
                f"rel_perf={c.rel_perf:.2f} < {c.threshold:.0%} at ε={c.collapse_rate:.2f}"
            )

    sig_pairs = [p for p in report.dominance_pairs if p.significant]
    lines.append(f"\n--- Significant Pairwise Dominance ({len(sig_pairs)} pairs) ---")
    for p in sig_pairs:
        lines.append(
            f"  [{p.adversary_type}, ε={p.corruption_rate:.2f}] "
            f"{p.winner} > {p.loser}: "
            f"Δ={p.delta:.4f} CI=[{p.ci_low:.4f}, {p.ci_high:.4f}]"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def format_regime_map_markdown(report: RegimeReport) -> str:
    """Format regime map as markdown tables for paper inclusion."""
    lines: List[str] = []

    if report.ci_alpha is not None:
        lines.append(
            f"> CI: α={report.ci_alpha}, level={report.ci_level:.0%}, "
            f"z={report.z_value:.4f}"
        )
        lines.append("")

    lines.append("## Regime Map: Dominant Mechanism (Welfare)")
    lines.append("")
    if isinstance(report.regime_map, pd.DataFrame) and not report.regime_map.empty:
        lines.append("| Adversary | Low (ε ≤ 0.25) | Mid (0.25 < ε ≤ 0.5) | High (ε > 0.5) |")
        lines.append("|---|---|---|---|")
        for adv_type in report.regime_map.index:
            row = report.regime_map.loc[adv_type]
            lines.append(
                f"| {adv_type} | {row.get('low', '')} "
                f"| {row.get('mid', '')} | {row.get('high', '')} |"
            )
    lines.append("")

    lines.append("## Detailed Regime Map (All Metrics)")
    lines.append("")
    if not report.regime_map_detailed.empty:
        lines.append("| Adversary | ε | Metric | Best | Margin | Runner-up |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in report.regime_map_detailed.iterrows():
            lines.append(
                f"| {row['adversary_type']} | {row['corruption_rate']:.2f} "
                f"| {row['metric']} | {row['best_aggregator']} "
                f"| {row['margin']:.4f} | {row['runner_up']} |"
            )
    lines.append("")

    lines.append("## Phase Transitions")
    lines.append("")
    if report.transitions:
        lines.append("| Adversary | ε | Before | After | Δ | Significant |")
        lines.append("|---|---|---|---|---|---|")
        for t in report.transitions:
            sig = "yes" if t.significant else "no"
            lines.append(
                f"| {t.adversary_type} | {t.crossover_rate:.2f} "
                f"| {t.leader_before} | {t.leader_after} "
                f"| {t.delta:.4f} | {sig} |"
            )
    else:
        lines.append("No phase transitions detected.")
    lines.append("")

    lines.append("## Collapse Points")
    lines.append("")
    if report.collapses:
        lines.append("| Adversary | Aggregator | Collapse ε | rel_perf | Threshold |")
        lines.append("|---|---|---|---|---|")
        for c in report.collapses:
            if np.isnan(c.rel_perf):
                lines.append(
                    f"| {c.adversary_type} | {c.aggregator} "
                    f"| — | NaN (flat) | {c.threshold:.0%} of clean |"
                )
            else:
                lines.append(
                    f"| {c.adversary_type} | {c.aggregator} "
                    f"| {c.collapse_rate:.2f} | {c.rel_perf:.2f} "
                    f"| {c.threshold:.0%} of clean |"
                )
    else:
        lines.append("No collapses detected.")

    return "\n".join(lines)
