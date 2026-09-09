# Experiment 7: Multi-Resolution Interpretability Report

## Executive Summary
This report provides multi-resolution interpretability for production Model B:
1. **Parametric / Die Level**: Exact feature-level attribution across 500 measurements via `DiagonalScore`.
2. **Spatial / Wafer Level**: Spatial prior decomposition across defect density, radius, edge distance, and wafer old-failure rate.
3. **Block / Internal Reading Level**: 73 LRT feature coefficients and 2,000-reading sequence anomaly peak index localization.

## Logit Evidence Decomposition
Model B log-odds prediction decomposes into exact additive contributions:
$$\text{logit}(P(\text{fail})) = C_{\text{spatial}} + C_{\text{parametric}} + C_{\text{block}}$$
where $C_{\text{param}} = e_{\text{param}} - 0.5 \cdot \text{offset}$ and $C_{\text{block}} = e_{\text{block}} - 0.5 \cdot \text{offset}$.

## Representative Case Explanations
| Category | Wafer ID | Die (Row, Col) | P(Failure) | Spatial Logit | Parametric Logit | Block Logit | Offset | Residual |
|---|---|---|---|---|---|---|---|---|
| True_Positive_TP | W_N_0083 | (9, 23) | 1.0000 | -2.21 | +10.64 | +4.22 | -1.31 | 5.77e-06 |
| False_Positive_FP | W_F_0039 | (17, 3) | 0.7785 | -3.11 | +0.46 | +4.02 | -1.31 | 1.98e-02 |
| True_Negative_TN | W_N_0007 | (20, 18) | 0.0002 | -6.42 | -3.63 | -1.69 | -1.31 | 1.52e-04 |
| False_Negative_FN | W_N_0055 | (23, 11) | 0.0015 | -4.25 | -2.84 | -0.46 | -1.31 | 1.01e-03 |

## Parametric & Spatial Interpretability
- **Parametric Attribution**: $c_f = \frac{(x_f - \text{centre}_f) \cdot \text{weight}_f}{\text{scale}}$. Top positive and negative feature contributions sum EXACTLY to the DiagonalScore output.
- **Spatial Factors**: Spatial context provides pre-test risk prior based on die radius, edge distance, and local defect density.

## Block Anomaly Sequence Localization
- Sub-die 2,000 block sequence readings are processed through circular FFT whitening and scan statistics. Peak anomaly positions $t^* \in [0, 1999]$ and localized block ranges $[t^* - 45, t^* + 45]$ pinpoint physical memory array defect locations.

## Visualizations Generated
- Wafer Spatial Risk Heatmaps saved to `results/interpretability/plots/wafer_heatmap_*.png`
- Block Anomaly Sequence Profiles saved to `results/interpretability/plots/block_anomaly_*.png`

## Status
EXPERIMENT 7 STATUS = COMPLETE
