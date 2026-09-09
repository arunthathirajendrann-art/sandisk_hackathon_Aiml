# Experiment 5: Adaptive / Gated Multi-Resolution Fusion

## Executive Summary
This experiment tests whether a small, interpretable 4-feature linear softmax gating model (`AdaptiveFusion`) that dynamically balances Parametric, Spatial, and Block LRT evidence can improve upon the current fixed Model B fusion.

## Benchmark Results (5-Fold Wafer-Grouped CV)
- **Model A Baseline**: AP = 0.582233, ROC-AUC = 0.896290, F1 = 0.550088
- **Model B Fixed Baseline**: AP = 0.653950, ROC-AUC = 0.925188, F1 = 0.604369
- **Model B Adaptive Gated**: AP = 0.598835, ROC-AUC = 0.896277, F1 = 0.467996

### Summary Table
| Experiment | AP | ROC-AUC | F1 | Precision | Recall | Δ AP vs Fixed B | Δ AP vs Model A | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| model_b_fixed | 0.653950 | 0.925188 | 0.604369 | 0.7525 | 0.5050 | +0.000000 | +0.071717 | 123.5s |
| model_b_equal | 0.601186 | 0.897057 | 0.562925 | 0.8565 | 0.4192 | -0.052764 | +0.018953 | 92.6s |
| model_b_adaptive | 0.598835 | 0.896277 | 0.467996 | 0.3768 | 0.6174 | -0.055115 | +0.016602 | 107.4s |

## Key Metric Deltas (Adaptive vs Fixed Model B)
- **Δ AP**: -0.055115
- **Δ ROC-AUC**: -0.028911
- **Δ F1 Score**: -0.136373

## Gate Weight Statistics
- **Average Weights**: Parametric = 0.4208, Spatial = 0.1607, Block = 0.4186
- **Distribution Range**:
  - Parametric: [0.1601, 0.7677]
  - Spatial: [0.0000, 0.7101]
  - Block: [0.1141, 0.7786]

## Final Decision
**EXPERIMENT 5 STATUS = REJECT**

### Decision Rationale
- Fixed Model B AP: 0.653950
- Adaptive Model B AP: 0.598835
- Delta AP: -0.055115
