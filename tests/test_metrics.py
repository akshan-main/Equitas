"""Test fairness, welfare, and robust statistics metrics."""
from __future__ import annotations

import numpy as np
import pytest

from equitas.metrics.fairness import (
    gini_coefficient,
    jain_fairness_index,
    unfairness_gap,
    worst_group_utility,
)
from equitas.metrics.robust_stats import bootstrap_ci, mad, trimmed_mean
from equitas.metrics.welfare import accuracy, city_utility, combined_loss, regret


class TestJainFairness:
    def test_perfect_equality(self):
        assert jain_fairness_index([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_max_inequality(self):
        # One class gets everything
        assert jain_fairness_index([1.0, 0.0, 0.0]) == pytest.approx(1 / 3, abs=0.01)

    def test_moderate(self):
        idx = jain_fairness_index([0.8, 0.5, 0.3])
        assert 0.0 < idx < 1.0

    def test_empty(self):
        assert jain_fairness_index([]) == 0.0


class TestWorstGroup:
    def test_basic(self):
        assert worst_group_utility({"a": 0.8, "b": 0.3, "c": 0.5}) == 0.3


class TestUnfairnessGap:
    def test_basic(self):
        assert unfairness_gap({"a": 0.8, "b": 0.3, "c": 0.5}) == pytest.approx(0.5)


class TestGini:
    def test_equality(self):
        assert gini_coefficient([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=0.01)

    def test_inequality(self):
        g = gini_coefficient([0.0, 0.0, 1.0])
        assert g > 0.5


class TestWelfare:
    def test_city_utility(self):
        assert city_utility({"a": 0.6, "b": 0.8}) == pytest.approx(0.7)

    def test_regret_positive(self):
        assert regret(0.5, 0.8) == pytest.approx(0.3)

    def test_regret_zero(self):
        assert regret(0.8, 0.8) == 0.0

    def test_combined_loss(self):
        loss = combined_loss(0.5, 0.8, 0.9, alpha=1.0, beta=0.5)
        expected = 1.0 * 0.3 + 0.5 * 0.1
        assert loss == pytest.approx(expected)

    def test_accuracy(self):
        assert accuracy("42", "42") == 1.0
        assert accuracy("42", "43") == 0.0


class TestTrimmedMean:
    def test_no_trim(self):
        assert trimmed_mean([1, 2, 3, 4, 5], alpha=0.0) == pytest.approx(3.0)

    def test_with_outliers(self):
        vals = [1, 2, 3, 4, 100]
        tm = trimmed_mean(vals, alpha=0.2)
        assert tm < np.mean(vals)


class TestMAD:
    def test_basic(self):
        assert mad([1, 2, 3, 4, 5]) == pytest.approx(1.0)


class TestBootstrap:
    def test_ci_contains_mean(self):
        vals = np.random.default_rng(0).normal(5, 1, 100)
        lo, hi = bootstrap_ci(vals, np.mean, n_boot=500)
        assert lo < 5.0 < hi


# --- Regime Analysis ---

import pandas as pd

from equitas.metrics.regime_analysis import (
    analyze_regimes,
    build_regime_map,
    find_collapse_points,
    find_crossover_points,
    format_regime_map_markdown,
    format_regime_report,
    mechanism_dominance_matrix,
    pairwise_ci_dominance,
)


def _make_sweep_summary() -> pd.DataFrame:
    """Synthetic sweep summary mimicking real sweep output."""
    rows = []
    # Selfish: MW dominates at high ε, majority_vote at low ε
    for rate in [0.0, 0.25, 0.5, 0.75]:
        # majority_vote degrades linearly
        mv_util = max(0.8 - rate * 0.8, 0.1)
        # MW stays robust
        mw_util = max(0.75 - rate * 0.2, 0.3)
        # EMA degrades moderately
        ema_util = max(0.78 - rate * 0.5, 0.2)
        for agg, util in [
            ("majority_vote", mv_util),
            ("multiplicative_weights", mw_util),
            ("ema_trust", ema_util),
        ]:
            rows.append({
                "corruption_rate": rate,
                "adversary_type": "selfish",
                "aggregator": agg,
                "trimmed_mean_utility": util,
                "mean_utility": util,
                "mean_fairness": 0.8,
                "trimmed_mean_fairness": 0.8,
                "ci_low_utility": util - 0.05,
                "ci_high_utility": util + 0.05,
                "ci_low_fairness": 0.75,
                "ci_high_fairness": 0.85,
                "mean_worst_group": util * 0.6,
                "mean_regret": 0.1,
                "ci_alpha": 0.1,
            })
    # Coordinated: MW always dominates
    for rate in [0.0, 0.25, 0.5, 0.75]:
        mv_util = max(0.8 - rate * 1.2, 0.05)
        mw_util = max(0.75 - rate * 0.3, 0.25)
        ema_util = max(0.78 - rate * 0.7, 0.1)
        for agg, util in [
            ("majority_vote", mv_util),
            ("multiplicative_weights", mw_util),
            ("ema_trust", ema_util),
        ]:
            rows.append({
                "corruption_rate": rate,
                "adversary_type": "coordinated",
                "aggregator": agg,
                "trimmed_mean_utility": util,
                "mean_utility": util,
                "mean_fairness": 0.7,
                "trimmed_mean_fairness": 0.7,
                "ci_low_utility": util - 0.05,
                "ci_high_utility": util + 0.05,
                "ci_low_fairness": 0.65,
                "ci_high_fairness": 0.75,
                "mean_worst_group": util * 0.5,
                "mean_regret": 0.15,
                "ci_alpha": 0.1,
            })
    return pd.DataFrame(rows)


class TestRegimeAnalysis:
    def test_crossover_points_with_significance(self):
        summary = _make_sweep_summary()
        transitions = find_crossover_points(summary)
        # Selfish: majority_vote leads at 0.0, MW takes over at some point
        selfish_transitions = [t for t in transitions if t.adversary_type == "selfish"]
        assert len(selfish_transitions) >= 1
        assert any(t.leader_after == "multiplicative_weights" for t in selfish_transitions)
        # Transitions have significance flag
        for t in transitions:
            assert isinstance(t.significant, bool)
            assert isinstance(t.delta, float)

    def test_collapse_points_with_rel_perf(self):
        summary = _make_sweep_summary()
        collapses = find_collapse_points(summary, threshold=0.5)
        # Majority vote should collapse under coordinated corruption
        coord_mv = [
            c for c in collapses
            if c.adversary_type == "coordinated" and c.aggregator == "majority_vote"
        ]
        assert len(coord_mv) >= 1
        assert coord_mv[0].collapse_rate <= 0.75
        # Collapse uses normalized rel_perf
        assert coord_mv[0].rel_perf < 0.5

    def test_pairwise_ci_dominance(self):
        summary = _make_sweep_summary()
        pairs = pairwise_ci_dominance(summary)
        assert len(pairs) > 0
        # Each pair has CI bounds
        for p in pairs:
            assert p.ci_low <= p.ci_high
            assert p.delta >= 0
            assert isinstance(p.significant, bool)

    def test_regime_map_multi_metric(self):
        summary = _make_sweep_summary()
        detailed, pivot = build_regime_map(summary)
        assert not pivot.empty
        assert not detailed.empty
        # At high corruption under coordinated, MW should win welfare
        assert pivot.loc["coordinated", "high"] == "multiplicative_weights"
        # Detailed has multiple metrics
        assert "welfare" in detailed["metric"].values
        assert "worst_group" in detailed["metric"].values
        # Detailed has margin and runner-up
        assert "margin" in detailed.columns
        assert "runner_up" in detailed.columns

    def test_dominance_matrix(self):
        summary = _make_sweep_summary()
        dom = mechanism_dominance_matrix(summary)
        assert "multiplicative_weights" in dom.index
        # MW should win most conditions under coordinated
        assert dom.loc["multiplicative_weights", "coordinated"] >= 0.5

    def test_full_analysis(self):
        summary = _make_sweep_summary()
        report = analyze_regimes(summary)
        assert not report.regime_map.empty
        assert not report.regime_map_detailed.empty
        assert isinstance(report.transitions, list)
        assert isinstance(report.collapses, list)
        assert isinstance(report.dominance_pairs, list)
        assert len(report.dominance_pairs) > 0
        # CI parameters stored in report
        assert report.ci_alpha == pytest.approx(0.1)
        assert report.ci_level == pytest.approx(0.9)
        assert report.z_value is not None
        assert report.z_value > 1.6  # z for 90% CI ≈ 1.645

    def test_format_report(self):
        summary = _make_sweep_summary()
        report = analyze_regimes(summary)
        text = format_regime_report(report)
        assert "REGIME ANALYSIS" in text
        assert "Phase Transitions" in text
        assert "Collapse Points" in text
        assert "Pairwise Dominance" in text
        # CI params printed
        assert "ci_alpha=" in text
        assert "ci_level=" in text
        assert "z=" in text

    def test_format_markdown(self):
        summary = _make_sweep_summary()
        report = analyze_regimes(summary)
        md = format_regime_map_markdown(report)
        assert "## Regime Map" in md
        assert "## Detailed Regime Map" in md
        assert "## Phase Transitions" in md
        assert "## Collapse Points" in md
        assert "|" in md  # Has tables
        # CI params in markdown header
        assert "level=90%" in md
