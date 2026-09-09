# Experiment 6: Probability Calibration Evaluation

## Executive Summary
This experiment evaluates whether Platt scaling (sigmoid calibration) or Isotonic regression improves the probability reliability (Brier Score, Log Loss, ECE) of production Model B without degrading ranking performance (AP, ROC-AUC).

## Benchmark Results (5-Fold Wafer-Grouped CV)
| Model | AP | ROC-AUC | F1 | Brier Score | Log Loss | ECE | Δ Brier | Δ LogLoss | Δ AP | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| model_b_raw | 0.653950 | 0.925188 | 0.604369 | 0.022105 | 0.089710 | 0.000897 | +0.000000 | +0.000000 | +0.000000 | BASELINE |
| model_b_platt | 0.653854 | 0.925163 | 0.604400 | 0.022110 | 0.089714 | 0.000892 | +0.000004 | +0.000003 | -0.000096 | **REJECT** |
| model_b_isotonic | 0.650889 | 0.924652 | 0.603530 | 0.022135 | 0.089868 | 0.000399 | +0.000030 | +0.000157 | -0.003061 | **REJECT** |

## Calibration Curve Summary (10 Bins)
- **Raw Model B ECE**: 0.000897
- **Platt Model B ECE**: 0.000892
- **Isotonic Model B ECE**: 0.000399

## Final Decision
**EXPERIMENT 6 STATUS = REJECT**

### Key Observations
- Platt Scaling Decision: **REJECT**
- Isotonic Regression Decision: **REJECT**
- **Production Integration Note**: Production Model B remains 100% untouched.
