# SanDisk Yield AI — Statistical Robustness Analysis Report (Phase 9 Step 4)

## Executive Summary
This report documents the wafer-level statistical robustness analysis evaluating whether Model B (Parametric + Spatial + 73 Sub-Die Block LRT Features) provides a statistically significant, stable performance improvement over Baseline Model A (Parametric + Spatial) across manufacturing wafers.

---

## 1. Methodology & Data Sources
- **Resampling Unit**: Complete Wafer (account for spatial autocorrelation within wafers).
- **Evaluation Dataset**: 160 Training/CV Wafers (154,037 eligible dies, `input/cache_train`).
- **Primary Metric**: Average Precision (AP / PR-AUC).
- **Secondary Metrics**: ROC-AUC, Failure F1 Score.
- **Statistical Tests**: Wafer-level Paired Wilcoxon Signed-Rank Test, Paired t-Test, Cohen's $d_z$ Effect Size.
- **Bootstrap Parameters**: 1,000 iterations resampling complete wafers with replacement (`seed=42`).

---

## 2. Key Statistical Findings

### Primary Metric: Average Precision (AP)
- **Observed ΔAP (Overall)**: `+0.073252` (`+12.59%` relative improvement)
- **Mean Per-Wafer ΔAP**: `+0.060280`
- **Median Per-Wafer ΔAP**: `+0.065924`
- **Wafer Consistency**: Model B outperforms Model A on **133 of 154 wafers** (**86.4%**).
- **Bootstrap 95% Confidence Interval**: `[0.067682, 0.078326]`
- **Paired Wilcoxon Signed-Rank Test**: Statistic = `450.0`, $p$-value = `1.373485e-22` (**Statistically Significant at $p < 0.05$**)
- **Paired t-Test**: $t$-statistic = `14.9530`, $p$-value = `9.885460e-32` (**Statistically Significant at $p < 0.05$**)
- **Effect Size (Cohen's $d_z$)**: `1.2049` (Large effect size (dz > 0.8))

### Secondary Metrics Summary
- **ROC-AUC**: Mean Per-Wafer ΔAUC = `+0.040704`, Wilcoxon $p$-value = `4.256100e-20`, Bootstrap 95% CI = `[0.025632, 0.033136]`.
- **Failure F1 Score**: Mean Per-Wafer ΔF1 = `+0.023663`, Wilcoxon $p$-value = `1.716516e-07`, Bootstrap 95% CI = `[0.045666, 0.064874]`.

---

## 3. Leakage & Safety Audit Checklist
- [x] **No Test Labels Used**: 0 competition test labels accessed.
- [x] **No Model Retraining**: Model weights & architecture untouched.
- [x] **No Threshold Tuning**: Established OOF decision thresholds preserved.
- [x] **Wafer-Level Resampling**: Bootstrap resamples whole wafers, avoiding die-level data leakage.
- [x] **Production Integrity**: `tuned/` and `submission.csv` strictly unmodified.

---

## 4. Conclusion
The statistical robustness analysis confirms that Model B's performance gain over Baseline Model A is **statistically robust, highly consistent across wafers, and non-random ($p < 10^-6$)**.
