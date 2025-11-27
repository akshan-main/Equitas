"""
Robust analysis layer for Equitas.

This is *offline* analysis that sits on top of the existing logs in results/.
It does NOT touch the core Kalilopolis / Equitas simulation.

Usage (from repo root):

    python -m backend.robust_eval

It expects that backend.experiments has already been run and that
results/aggregator_log.csv and results/agent_log.csv exist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
AGG_LOG = os.path.join(RESULTS_DIR, "aggregator_log.csv")
AGENT_LOG = os.path.join(RESULTS_DIR, "agent_log.csv")


@dataclass
class SpectralFilterResult:
    judge_epsilon: float
    advisor_epsilon: float
    run: int
    judge_indices: List[int]
    scores: np.ndarray
    keep_mask: np.ndarray
    threshold: float


def _load_logs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(AGG_LOG)
    agents = pd.read_csv(AGENT_LOG)
    return agg, agents


def spectral_filter_judges(
    agents: pd.DataFrame,
    judge_epsilon: float,
    advisor_epsilon: float,
    run: int | None = None,
    tau: float = 2.5,
) -> SpectralFilterResult:
    """Detect suspicious judges via a simple spectral signature test.

    We build a matrix J x C where J = #judges, C = #crises, with entries
    equal to the per-judge per-crisis *loss*. We centre columns and
    compute the covariance across judges, then take the top eigenvector.
    Judges with large absolute coordinates on this eigenvector are
    treated as potential outliers.
    """
    df = agents[
        (agents["panel"] == "judge")
        & (agents["judge_epsilon"] == judge_epsilon)
        & (agents["advisor_epsilon"] == advisor_epsilon)
    ]
    if run is not None:
        df = df[df["run"] == run]

    judges = sorted(df["agent_index"].unique())
    crises = sorted(df["crisis_id"].unique())
    J = len(judges)
    C = len(crises)

    loss_mat = np.zeros((J, C), dtype=float)
    loss_mat[:] = np.nan

    for j_idx, j in enumerate(judges):
        for c_idx, c in enumerate(crises):
            sub = df[(df["agent_index"] == j) & (df["crisis_id"] == c)]
            if sub.empty:
                continue
            loss_mat[j_idx, c_idx] = float(sub["loss"].iloc[0])

    # Fill missing entries with column means
    col_means = np.nanmean(loss_mat, axis=0)
    inds = np.where(np.isnan(loss_mat))
    loss_mat[inds] = np.take(col_means, inds[1])

    # Centre per crisis
    loss_centered = loss_mat - loss_mat.mean(axis=0, keepdims=True)

    # Covariance over judges (rows)
    cov = loss_centered @ loss_centered.T / max(C, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]  # top eigenvector
    scores = np.abs(v)

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    if mad == 0.0:
        mad = float(scores.std() + 1e-8)

    threshold = med + tau * mad
    keep_mask = scores <= threshold

    return SpectralFilterResult(
        judge_epsilon=judge_epsilon,
        advisor_epsilon=advisor_epsilon,
        run=int(df["run"].iloc[0]) if not df.empty else 0,
        judge_indices=judges,
        scores=scores,
        keep_mask=keep_mask,
        threshold=threshold,
    )


def _recompute_equal_aggregator(
    agents: pd.DataFrame,
    judge_epsilon: float,
    advisor_epsilon: float,
    run: int | None,
    keep_judges: List[int] | None = None,
    name: str = "judge_equal_spectral",
) -> pd.DataFrame:
    """Recompute an equal-weight vote over judges, optionally after
    dropping a subset of judges.

    This is purely offline: it uses logged judge recommendations +
    the world-evaluated metrics for those recommendations.
    """
    df = agents[
        (agents["panel"] == "judge")
        & (agents["judge_epsilon"] == judge_epsilon)
        & (agents["advisor_epsilon"] == advisor_epsilon)
    ]
    if run is not None:
        df = df[df["run"] == run]
    if keep_judges is not None:
        df = df[df["agent_index"].isin(keep_judges)]

    rows: List[Dict] = []
    for crisis_id in sorted(df["crisis_id"].unique()):
        sub = df[df["crisis_id"] == crisis_id]
        if sub.empty:
            continue

        # equal-weight majority vote with deterministic tie-breaking
        counts = sub["recommended_action_id"].value_counts()
        best_action = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        chosen_rows = sub[sub["recommended_action_id"] == best_action]
        r0 = chosen_rows.iloc[0]

        rows.append(
            {
                "crisis_id": crisis_id,
                "aggregator": name,
                "chosen_action_id": best_action,
                "city_utility": float(r0["adv_city_utility"]),
                "unfairness_gap": float(r0["adv_unfairness_gap"]),
                "fairness_jain": float(r0["adv_fairness_jain"]),
                "advisor_epsilon": advisor_epsilon,
                "judge_epsilon": judge_epsilon,
                "run": int(r0["run"]),
            }
        )

    return pd.DataFrame(rows)


def run_robust_analysis(
    tau: float = 2.5,
    advisor_epsilon: float | None = None,
) -> None:
    """Top-level entry point.

    For each (judge_epsilon, run), we:
      1) perform spectral filtering to identify suspicious judges;
      2) recompute an equal-weight judge aggregator after removing them;
      3) save a new aggregator_log_robust.csv and a judge_diagnostics.csv.
    """
    agg, agents = _load_logs()

    if advisor_epsilon is None:
        advisor_vals = sorted(agg["advisor_epsilon"].unique())
    else:
        advisor_vals = [advisor_epsilon]

    robust_agg_dfs: List[pd.DataFrame] = []
    diag_rows: List[Dict] = []

    for a_eps in advisor_vals:
        sub_aggs = agg[agg["advisor_epsilon"] == a_eps]
        for j_eps in sorted(sub_aggs["judge_epsilon"].unique()):
            for run in sorted(sub_aggs["run"].unique()):
                # 1) spectral filter
                sf = spectral_filter_judges(
                    agents,
                    judge_epsilon=j_eps,
                    advisor_epsilon=a_eps,
                    run=run,
                    tau=tau,
                )
                keep_judges = [
                    j for j, keep in zip(sf.judge_indices, sf.keep_mask) if keep
                ]

                for j, score, keep in zip(sf.judge_indices, sf.scores, sf.keep_mask):
                    diag_rows.append(
                        {
                            "advisor_epsilon": a_eps,
                            "judge_epsilon": j_eps,
                            "run": run,
                            "agent_index": j,
                            "spectral_score": float(score),
                            "keep": bool(keep),
                        }
                    )

                # 2) recompute filtered equal aggregator
                robust_agg = _recompute_equal_aggregator(
                    agents,
                    judge_epsilon=j_eps,
                    advisor_epsilon=a_eps,
                    run=run,
                    keep_judges=keep_judges,
                    name="judge_equal_spectral",
                )
                robust_agg_dfs.append(robust_agg)

    robust_agg_all = pd.concat(robust_agg_dfs, ignore_index=True)
    diag_df = pd.DataFrame(diag_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    robust_path = os.path.join(RESULTS_DIR, "aggregator_log_robust.csv")
    diag_path = os.path.join(RESULTS_DIR, "judge_diagnostics.csv")

    robust_agg_all.to_csv(robust_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    print(f"Saved robust aggregator log to {robust_path}")
    print(f"Saved judge diagnostics to {diag_path}")


def main():
    run_robust_analysis()


if __name__ == "__main__":
    main()
