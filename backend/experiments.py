"""
Experiment runner:

- Sweeps epsilon (fraction of corrupted advisors).
- Runs multiple random seeds per epsilon.
- Compares equal-weight vs MW aggregator on:
    - city utility
    - unfairness gap
    - Jain fairness index
- Uses robust stats (trimmed mean) + bootstrap CI.

This is the script you'd run for your robust stats / governance experiments.
"""

from __future__ import annotations

import os
import asyncio
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .simulate import SimulationConfig, run_simulation
from .robust_stats import trimmed_mean, bootstrap_ci


# You can tweak these directly for different experiment regimes
EPSILONS = [0.0, 0.25, 0.5, 0.75]
NUM_RUNS_PER_EPSILON = 3
NUM_CRISES = 15  # bump this for “paper-scale” experiments


async def run_experiments_async():
    all_agg: List[pd.DataFrame] = []
    all_adv: List[pd.DataFrame] = []

    seed_base = 2025

    for eps_idx, eps in enumerate(EPSILONS):
        for run_idx in range(NUM_RUNS_PER_EPSILON):
            seed = seed_base + eps_idx * 100 + run_idx
            print(f"Running epsilon={eps:.2f}, run={run_idx}, seed={seed}...")
            config = SimulationConfig(
                num_crises=NUM_CRISES,
                epsilon_corruption=eps,
                seed=seed,
                eta=1.0,
                alpha=1.0,
                beta=0.5,
            )
            result = await run_simulation(config)
            agg = result.aggregator_log.copy()
            adv = result.advisor_log.copy()
            agg["run"] = run_idx
            adv["run"] = run_idx
            all_agg.append(agg)
            all_adv.append(adv)

    agg_all = pd.concat(all_agg, ignore_index=True)
    adv_all = pd.concat(all_adv, ignore_index=True)

    # === Robust summary stats per epsilon x aggregator ===

    summary_rows = []
    rng = np.random.default_rng(42)

    for eps in EPSILONS:
        for agg_name in ["equal", "mw"]:
            sub = agg_all[
                (agg_all["epsilon"] == eps)
                & (agg_all["aggregator"] == agg_name)
            ]
            util = sub["city_utility"].values
            unfair_gap = sub["unfairness_gap"].values
            fairness_j = sub["fairness_jain"].values

            if util.size == 0:
                continue

            # City utility stats
            mean_util = float(util.mean())
            med_util = float(np.median(util))
            tmean_util = trimmed_mean(util, alpha=0.1)
            ci_low_u, ci_high_u = bootstrap_ci(
                util, estimator=lambda x: trimmed_mean(x, alpha=0.1),
                n_boot=500, alpha=0.10, rng=rng,
            )

            # Unfairness gap stats (lower is better)
            mean_unfair = float(unfair_gap.mean())
            med_unfair = float(np.median(unfair_gap))
            tmean_unfair = trimmed_mean(unfair_gap, alpha=0.1)

            # Jain fairness stats (higher is better)
            mean_fair_j = float(fairness_j.mean())
            med_fair_j = float(np.median(fairness_j))
            tmean_fair_j = trimmed_mean(fairness_j, alpha=0.1)
            ci_low_f, ci_high_f = bootstrap_ci(
                fairness_j, estimator=lambda x: trimmed_mean(x, alpha=0.1),
                n_boot=500, alpha=0.10, rng=rng,
            )

            summary_rows.append(
                {
                    "epsilon": eps,
                    "aggregator": agg_name,
                    "mean_util": mean_util,
                    "median_util": med_util,
                    "trimmed_mean_util": tmean_util,
                    "trimmed_mean_util_ci_low": ci_low_u,
                    "trimmed_mean_util_ci_high": ci_high_u,
                    "mean_unfair_gap": mean_unfair,
                    "median_unfair_gap": med_unfair,
                    "trimmed_mean_unfair_gap": tmean_unfair,
                    "mean_fairness_jain": mean_fair_j,
                    "median_fairness_jain": med_fair_j,
                    "trimmed_mean_fairness_jain": tmean_fair_j,
                    "trimmed_mean_fairness_jain_ci_low": ci_low_f,
                    "trimmed_mean_fairness_jain_ci_high": ci_high_f,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Robust summary (city utility & fairness) ===")
    print(summary_df)

    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/summary.csv", index=False)
    agg_all.to_csv("results/aggregator_log.csv", index=False)
    adv_all.to_csv("results/advisor_log.csv", index=False)
    print("\nSaved raw logs and summary to results/")

    # === Plots ===

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # City utility vs epsilon
    for agg_name, marker in [("equal", "o"), ("mw", "s")]:
        sub = summary_df[summary_df["aggregator"] == agg_name]
        if sub.empty:
            continue
        axes[0].plot(
            sub["epsilon"],
            sub["trimmed_mean_util"],
            marker + "-",
            label=f"{agg_name} (trimmed mean)",
        )
        axes[0].fill_between(
            sub["epsilon"],
            sub["trimmed_mean_util_ci_low"],
            sub["trimmed_mean_util_ci_high"],
            alpha=0.2,
        )

    axes[0].set_xlabel("Corruption fraction ε")
    axes[0].set_ylabel("City utility (trimmed mean)")
    axes[0].set_title("City welfare vs corruption")
    axes[0].legend()

    # Jain fairness vs epsilon
    for agg_name, marker in [("equal", "o"), ("mw", "s")]:
        sub = summary_df[summary_df["aggregator"] == agg_name]
        if sub.empty:
            continue
        axes[1].plot(
            sub["epsilon"],
            sub["trimmed_mean_fairness_jain"],
            marker + "-",
            label=f"{agg_name} (trimmed mean)",
        )
        axes[1].fill_between(
            sub["epsilon"],
            sub["trimmed_mean_fairness_jain_ci_low"],
            sub["trimmed_mean_fairness_jain_ci_high"],
            alpha=0.2,
        )

    axes[1].set_xlabel("Corruption fraction ε")
    axes[1].set_ylabel("Fairness (Jain index, trimmed mean)")
    axes[1].set_title("Fairness vs corruption")
    axes[1].legend()

    plt.tight_layout()
    out_path = "results/performance_vs_epsilon.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    asyncio.run(run_experiments_async())


if __name__ == "__main__":
    main()
