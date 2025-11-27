# Equitas

## Multiplicative Weights Aggregation for Robust LLM Governance

This repo simulates a Plato-style city governed by multiple LLM advisors (one per social class),
some of which are corrupted. A central governor uses:

- **Equal-weight voting** (baseline)
- **Multiplicative Weights (MW)** over advisors, with losses based on:
  - City utility (welfare)
  - Fairness (Jain index)

We log outcomes and compute robust statistics over corruption levels.
