---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - benchmark
  - robustness
  - multi-agent
  - fairness
  - llm
  - corruption
  - aggregation
pretty_name: "Equitas: Corruption-Robustness Benchmark for Multi-LLM Committees"
size_categories:
  - 1K<n<10K
---

# Equitas: A Corruption-Robustness Benchmark for Multi-LLM Committees

## Overview

Equitas is a benchmark for evaluating aggregation strategies in hierarchical multi-LLM committees under adversarial corruption. It measures how well different aggregation methods maintain **utility** (task performance) and **fairness** (equitable outcomes across stakeholder groups) when a fraction of committee members are corrupted by adversaries.

All experiments use **gpt-4o-mini** as the underlying LLM through a simulated governance task (Plato's city with three citizen classes: guardians, auxiliaries, producers).

## What This Dataset Contains

### Benchmark Tables (`tables/`)

15 result tables (CSV format) from the full experiment suite:

| File | Description |
|------|-------------|
| `B1_aggregator_leaderboard.csv` | Overall ranking of 10 aggregators by utility, fairness, worst-group utility, and regret |
| `B2_utility_by_corruption.csv` | Utility at corruption rates ε ∈ {0.00, 0.25, 0.50, 0.75} with robustness ratios |
| `B3_utility_by_adversary.csv` | Utility broken down by 4 adversary types |
| `B4_regime_winners_welfare.csv` | Best aggregator per (ε, adversary) cell for welfare |
| `B4b_regime_winners_fairness.csv` | Best aggregator per cell for Jain fairness |
| `B4b_regime_winners_worst_group.csv` | Best aggregator per cell for worst-group utility |
| `B5_recovery.csv` | Recovery after mid-run corruption onset at round 20/40 |
| `B6_scaling.csv` | Utility and fairness vs. committee size (N ∈ {3,5,7,10,15}) |
| `B7_hier_vs_flat.csv` | Hierarchical vs. flat architecture comparison |
| `B7b_hier_vs_flat_detail_075.csv` | Architecture comparison detail at ε=0.75 |
| `B8_pareto_mw.csv` | MW Pareto sweep over (α, β) welfare-fairness tradeoff |
| `B8b_pareto_frontier_points.csv` | Pareto-optimal points from the frontier |
| `D1_go_vs_fh_gap.csv` | Governor-only vs. full-hierarchy protocol gap |
| `D1b_go_vs_fh_high_corruption.csv` | Protocol comparison at high corruption |
| `D2_go_vs_fh_grand_summary.csv` | Grand summary across all experiment types |

### Experiment Configs (`configs/`)

13 YAML configuration files specifying exact parameters for each experiment (corruption rates, adversary types, committee sizes, number of runs, etc.). These enable full reproducibility.

### Figures (`figures/`)

6 paper-quality PNG plots:
- `fig_corruption_sweep.png` — Utility vs. corruption rate by adversary type
- `fig_fairness_sweep.png` — Jain fairness vs. corruption rate
- `fig_worst_group.png` — Worst-group utility vs. corruption rate
- `fig_recovery.png` — Recovery trajectory and MW weight evolution
- `fig_scaling.png` — Utility and fairness vs. committee size
- `fig_pareto.png` — Welfare-fairness Pareto frontier

## Experimental Setup

### Aggregation Methods (10 total)

| Method | Description |
|--------|-------------|
| Oracle | Hindsight-optimal: picks the action that actually maximized city utility each round |
| MW (Multiplicative Weights) | Updates agent weights via w *= exp(-η·loss); hierarchical at both levels |
| Supervisor | LLM re-ranks proposals using judge feedback |
| Confidence-Weighted | Weights votes by self-reported confidence |
| EMA Trust | Exponential moving average of past accuracy as weights |
| Trimmed Vote | Drops top-20% loss agents, then majority vote |
| Majority Vote | Equal-weight plurality, no learning |
| Self-Consistency | Same LLM sampled K=5 times, majority vote |
| Oracle Upper Bound | Best-of-K oracle (theoretical ceiling for self-consistency) |
| Random Dictator | Picks one agent uniformly at random each round |

### Adversary Types (4)

| Type | Behavior |
|------|----------|
| Selfish | Maximizes own class utility using world model |
| Coordinated | All corrupted agents push the same wrong action |
| Scheduled | Honest for first half to build trust, then exploits |
| Deceptive | Picks wrong action with persuasive LLM-generated rationale |

### Corruption Rates

ε ∈ {0.00, 0.25, 0.50, 0.75} — fraction of agents replaced by adversaries.

### Protocol

Each experiment runs 40 rounds. Results are averaged over 3 independent runs with different random seeds. Confidence intervals use bootstrap resampling.

**Hierarchical (Full-Hierarchy) protocol:**
1. 7 members per class → intra-class aggregation → 1 leader per class
2. 3 leaders produce proposals → 5 judges evaluate → inter-class aggregation
3. Governor (algorithmic, no LLM) selects final action

## Key Observations From the Data

> These are observations, not claims of superiority. Confidence intervals overlap for most non-oracle methods.

- **Oracle** achieves 0.4655 mean utility — a soft ceiling since it uses hindsight information unavailable in practice.
- **Supervisor** shows the highest robustness ratio (0.996) — its utility at ε=0.75 is 99.6% of its ε=0.00 value. MW's robustness ratio is 0.955.
- At **low corruption** (ε ≤ 0.50), most methods perform similarly (utility spread < 0.01).
- At **high corruption** (ε = 0.75), methods diverge: supervisor maintains 0.4475 while majority vote drops to 0.3997.
- **Confidence intervals overlap** for most non-oracle methods across conditions, so ranking differences should be interpreted cautiously.
- **Hierarchical architecture** outperforms flat at ε=0.75 (utility gap +0.049), but the two are comparable at low corruption.
- **Scaling** shows an inverted-U: performance peaks around N=7 members per class, then degrades as committee size increases.

## Limitations

- **Single LLM**: All experiments use gpt-4o-mini. Results may not generalize to other models or heterogeneous committees.
- **Single task domain**: The governance simulation is stylized. Real-world multi-agent tasks may behave differently.
- **Moderate scale**: 40 rounds, 3 runs per condition. Larger-scale experiments might reveal different patterns.
- **No human evaluation**: All judgments are LLM-generated; no human ground truth for the governance task itself (utility is computed from a known sigmoid function).

## Citation

If you use this dataset, please cite:

```bibtex
@misc{equitas2025,
  title={Equitas: A Corruption-Robustness Benchmark for Hierarchical Multi-LLM Committees},
  author={Krithick, Akshan},
  year={2026},
  url={https://huggingface.co/datasets/akshan-main/Equitas}
}
```

## License

MIT
