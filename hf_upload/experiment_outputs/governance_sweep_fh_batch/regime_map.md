> CI: α=0.1, level=90%, z=1.6452

## Regime Map: Dominant Mechanism (Welfare)

| Adversary | Low (ε ≤ 0.25) | Mid (0.25 < ε ≤ 0.5) | High (ε > 0.5) |
|---|---|---|---|
| coordinated | oracle | oracle | oracle |
| deceptive | oracle | oracle | oracle |
| scheduled | oracle | oracle | oracle |
| selfish | oracle | oracle | oracle |

## Detailed Regime Map (All Metrics)

| Adversary | ε | Metric | Best | Margin | Runner-up |
|---|---|---|---|---|---|
| coordinated | 0.00 | welfare | oracle | 0.0136 | self_consistency |
| coordinated | 0.25 | welfare | oracle | 0.0129 | self_consistency |
| coordinated | 0.50 | welfare | oracle | 0.0132 | supervisor_rerank |
| coordinated | 0.75 | welfare | oracle | 0.0139 | supervisor_rerank |
| deceptive | 0.00 | welfare | oracle | 0.0118 | random_dictator |
| deceptive | 0.25 | welfare | oracle | 0.0203 | trimmed_vote |
| deceptive | 0.50 | welfare | oracle | 0.0189 | multiplicative_weights |
| deceptive | 0.75 | welfare | oracle | 0.0168 | supervisor_rerank |
| scheduled | 0.00 | welfare | oracle | 0.0122 | ema_trust |
| scheduled | 0.25 | welfare | oracle | 0.0176 | ema_trust |
| scheduled | 0.50 | welfare | oracle | 0.0145 | supervisor_rerank |
| scheduled | 0.75 | welfare | oracle | 0.0201 | supervisor_rerank |
| selfish | 0.00 | welfare | oracle | 0.0155 | self_consistency |
| selfish | 0.25 | welfare | oracle | 0.0121 | trimmed_vote |
| selfish | 0.50 | welfare | oracle | 0.0116 | ema_trust |
| selfish | 0.75 | welfare | oracle | 0.0097 | confidence_weighted |
| coordinated | 0.00 | worst_group | self_consistency | 0.0008 | confidence_weighted |
| coordinated | 0.25 | worst_group | trimmed_vote | 0.0001 | majority_vote |
| coordinated | 0.50 | worst_group | confidence_weighted | 0.0008 | trimmed_vote |
| coordinated | 0.75 | worst_group | supervisor_rerank | 0.0121 | oracle |
| deceptive | 0.00 | worst_group | majority_vote | 0.0000 | confidence_weighted |
| deceptive | 0.25 | worst_group | majority_vote | 0.0001 | trimmed_vote |
| deceptive | 0.50 | worst_group | multiplicative_weights | 0.0005 | trimmed_vote |
| deceptive | 0.75 | worst_group | supervisor_rerank | 0.0115 | oracle |
| scheduled | 0.00 | worst_group | ema_trust | 0.0001 | trimmed_vote |
| scheduled | 0.25 | worst_group | ema_trust | 0.0005 | confidence_weighted |
| scheduled | 0.50 | worst_group | trimmed_vote | 0.0002 | multiplicative_weights |
| scheduled | 0.75 | worst_group | supervisor_rerank | 0.0072 | oracle |
| selfish | 0.00 | worst_group | self_consistency | 0.0003 | trimmed_vote |
| selfish | 0.25 | worst_group | confidence_weighted | 0.0000 | multiplicative_weights |
| selfish | 0.50 | worst_group | trimmed_vote | 0.0000 | majority_vote |
| selfish | 0.75 | worst_group | trimmed_vote | 0.0010 | confidence_weighted |
| coordinated | 0.00 | fairness | self_consistency | 0.0002 | confidence_weighted |
| coordinated | 0.25 | fairness | majority_vote | 0.0002 | trimmed_vote |
| coordinated | 0.50 | fairness | ema_trust | 0.0007 | majority_vote |
| coordinated | 0.75 | fairness | supervisor_rerank | 0.0027 | multiplicative_weights |
| deceptive | 0.00 | fairness | confidence_weighted | 0.0000 | ema_trust |
| deceptive | 0.25 | fairness | majority_vote | 0.0004 | confidence_weighted |
| deceptive | 0.50 | fairness | trimmed_vote | 0.0001 | confidence_weighted |
| deceptive | 0.75 | fairness | supervisor_rerank | 0.0015 | multiplicative_weights |
| scheduled | 0.00 | fairness | random_dictator | 0.0002 | trimmed_vote |
| scheduled | 0.25 | fairness | ema_trust | 0.0003 | confidence_weighted |
| scheduled | 0.50 | fairness | ema_trust | 0.0000 | trimmed_vote |
| scheduled | 0.75 | fairness | supervisor_rerank | 0.0036 | random_dictator |
| selfish | 0.00 | fairness | self_consistency | 0.0001 | majority_vote |
| selfish | 0.25 | fairness | self_consistency | 0.0000 | confidence_weighted |
| selfish | 0.50 | fairness | supervisor_rerank | 0.0013 | trimmed_vote |
| selfish | 0.75 | fairness | trimmed_vote | 0.0005 | random_dictator |

## Phase Transitions

No phase transitions detected.

## Collapse Points

| Adversary | Aggregator | Collapse ε | rel_perf | Threshold |
|---|---|---|---|---|
| coordinated | confidence_weighted | 0.75 | 0.00 | 50% of clean |
| coordinated | ema_trust | 0.75 | 0.00 | 50% of clean |
| coordinated | majority_vote | 0.75 | 0.00 | 50% of clean |
| coordinated | multiplicative_weights | 0.75 | 0.00 | 50% of clean |
| coordinated | oracle | 0.75 | 0.00 | 50% of clean |
| coordinated | random_dictator | 0.75 | 0.00 | 50% of clean |
| coordinated | self_consistency | 0.75 | 0.00 | 50% of clean |
| coordinated | supervisor_rerank | 0.75 | 0.00 | 50% of clean |
| coordinated | trimmed_vote | 0.75 | 0.00 | 50% of clean |
| deceptive | confidence_weighted | 0.75 | 0.00 | 50% of clean |
| deceptive | ema_trust | 0.75 | 0.00 | 50% of clean |
| deceptive | majority_vote | 0.75 | 0.00 | 50% of clean |
| deceptive | multiplicative_weights | 0.75 | 0.00 | 50% of clean |
| deceptive | oracle | — | NaN (flat) | 50% of clean |
| deceptive | random_dictator | 0.50 | 0.48 | 50% of clean |
| deceptive | self_consistency | 0.75 | 0.00 | 50% of clean |
| deceptive | supervisor_rerank | — | NaN (flat) | 50% of clean |
| deceptive | trimmed_vote | 0.75 | 0.00 | 50% of clean |
| scheduled | confidence_weighted | 0.75 | 0.00 | 50% of clean |
| scheduled | ema_trust | 0.75 | 0.00 | 50% of clean |
| scheduled | majority_vote | 0.75 | 0.00 | 50% of clean |
| scheduled | multiplicative_weights | 0.75 | 0.00 | 50% of clean |
| scheduled | oracle | 0.75 | 0.00 | 50% of clean |
| scheduled | random_dictator | 0.75 | 0.00 | 50% of clean |
| scheduled | self_consistency | 0.75 | 0.00 | 50% of clean |
| scheduled | supervisor_rerank | 0.75 | 0.00 | 50% of clean |
| scheduled | trimmed_vote | 0.75 | 0.00 | 50% of clean |
| selfish | confidence_weighted | 0.25 | 0.00 | 50% of clean |
| selfish | ema_trust | 0.25 | 0.00 | 50% of clean |
| selfish | majority_vote | 0.25 | 0.00 | 50% of clean |
| selfish | multiplicative_weights | 0.25 | 0.00 | 50% of clean |
| selfish | oracle | 0.25 | 0.00 | 50% of clean |
| selfish | random_dictator | 0.25 | 0.00 | 50% of clean |
| selfish | self_consistency | 0.25 | 0.00 | 50% of clean |
| selfish | supervisor_rerank | 0.25 | 0.00 | 50% of clean |
| selfish | trimmed_vote | 0.25 | 0.00 | 50% of clean |