# Experiment 4: Block Encoder Comparison

## Executive Summary
This experiment compares three block encoder architectures on 2,000 sub-die block readings:
1. **LRT**: Statistical circular FFT-whitened likelihood-ratio scan statistics (~73 features)
2. **CNN**: 1D Convolutional Neural Network trained directly on 2,000-length sequence readings
3. **LRT + CNN**: Combined block representation fusing LRT likelihood ratios and 1D CNN representations

## Benchmark Results (5-Fold Wafer-Grouped CV)
- **Model A Baseline**: AP = 0.582233, ROC-AUC = 0.896290, F1 = 0.550088

### Summary Metrics Table
| Experiment | AP | ROC-AUC | F1 | Precision | Recall | Δ AP vs LRT | Δ AP vs Model A | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| block_only_LRT | 0.159130 | 0.744907 | 0.221748 | 0.191723 | 0.262924 | +0.000000 | -0.423103 | 66.8s |
| block_only_CNN | 0.081419 | 0.669299 | 0.147351 | 0.098653 | 0.290996 | -0.077712 | -0.500814 | 46.4s |
| block_only_LRT_CNN | 0.105607 | 0.701763 | 0.178028 | 0.135519 | 0.259396 | -0.053523 | -0.476626 | 67.2s |
| model_b_LRT | 0.654872 | 0.925632 | 0.605060 | 0.728766 | 0.517257 | +0.000000 | +0.072639 | 89.7s |
| model_b_CNN | 0.575947 | 0.893619 | 0.553960 | 0.762366 | 0.435036 | -0.078925 | -0.006286 | 70.8s |
| model_b_LRT_CNN | 0.589445 | 0.901185 | 0.561636 | 0.750577 | 0.448688 | -0.065427 | +0.007212 | 89.0s |

## Key Findings & Comparisons

1. **CNN vs LRT Standalone (`block_only`)**:
   - `block_only_LRT`: AP = 0.159130
   - `block_only_CNN`: AP = 0.081419
   - Delta (CNN - LRT): -0.077712

2. **CNN vs LRT in Model B Context**:
   - `model_b_LRT`: AP = 0.654872
   - `model_b_CNN`: AP = 0.575947
   - Delta (Model B CNN - Model B LRT): -0.078925

3. **LRT + CNN Combination**:
   - `model_b_LRT_CNN`: AP = 0.589445
   - Delta (LRT+CNN vs LRT): -0.065427

## Recommendation
- **REJECT CNN**. The existing statistical LRT block encoder outperforms the 1D CNN and LRT+CNN combination while being computationally lighter and domain-derived.
- **Retain statistical LRT block encoder as the candidate for Model B**.
