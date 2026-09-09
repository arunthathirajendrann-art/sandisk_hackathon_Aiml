# SanDisk Yield AI — Failure Signature Discovery Analysis (Phase 9 Step 5)

## Executive Summary
This exploratory analysis investigates whether die-level failures across the manufacturing dataset group into recurring multi-resolution evidence signatures using Model B's evidence decomposition (Parametric, Spatial, and Block LRT branches).

---

## 1. Input Representation & Methodology
- **Dataset**: 160 Training/CV Wafers (154,037 eligible dies, `input/cache_train`).
- **Evidence Dimensions**:
  1. `parametric_evidence` ($z_{	ext{param}}$)
  2. `spatial_evidence` ($z_{	ext{spatial}}$)
  3. `block_evidence` ($z_{	ext{block}}$)
  4. `risk_difference` ($\Delta P = P_B - P_A$)
- **Preprocessing**: `StandardScaler` fitted strictly on the analysis dataset ($154,037$ dies). Zero labels used during clustering.
- **Clustering Method**: `HDBSCAN`.

---

## 2. Discovered Failure Signature Clusters

### Cluster 0: Spatial-dominant (Spatial Dominant)
- **Die Count**: `62,740` (`40.73%` of analyzed dies)
- **Observed Failures (Post-hoc Descriptive)**: `818` (Failure Rate: `1.30%`)
- **Mean Model A Probability**: `0.0280`
- **Mean Model B Probability**: `0.0125`
- **Mean Risk Difference ($\Delta P$)**: `+-0.0155` if row.mean_risk_difference >= 0 else `-0.0155`
- **Evidence Profile**: Parametric = `-0.470`, Spatial = `0.755`, Block LRT = `-2.836`

### Cluster 1: Block-dominant (Block LRT Dominant)
- **Die Count**: `2,668` (`1.73%` of analyzed dies)
- **Observed Failures (Post-hoc Descriptive)**: `1,039` (Failure Rate: `38.94%`)
- **Mean Model A Probability**: `0.1424`
- **Mean Model B Probability**: `0.3862`
- **Mean Risk Difference ($\Delta P$)**: `+0.2437` if row.mean_risk_difference >= 0 else `0.2437`
- **Evidence Profile**: Parametric = `0.874`, Spatial = `0.681`, Block LRT = `-0.101`

### Cluster 2: Low-evidence / uncertain (Low-evidence Dominant)
- **Die Count**: `33,810` (`21.95%` of analyzed dies)
- **Observed Failures (Post-hoc Descriptive)**: `574` (Failure Rate: `1.70%`)
- **Mean Model A Probability**: `0.0280`
- **Mean Model B Probability**: `0.0193`
- **Mean Risk Difference ($\Delta P$)**: `+-0.0087` if row.mean_risk_difference >= 0 else `-0.0087`
- **Evidence Profile**: Parametric = `-0.425`, Spatial = `0.358`, Block LRT = `-2.305`

### Cluster 3: Block-dominant (Block LRT Dominant)
- **Die Count**: `52,514` (`34.09%` of analyzed dies)
- **Observed Failures (Post-hoc Descriptive)**: `1,789` (Failure Rate: `3.41%`)
- **Mean Model A Probability**: `0.0218`
- **Mean Model B Probability**: `0.0341`
- **Mean Risk Difference ($\Delta P$)**: `+0.0123` if row.mean_risk_difference >= 0 else `0.0123`
- **Evidence Profile**: Parametric = `-0.582`, Spatial = `0.727`, Block LRT = `-1.360`

### Cluster 4: Parametric-dominant (Parametric Dominant)
- **Die Count**: `2,305` (`1.50%` of analyzed dies)
- **Observed Failures (Post-hoc Descriptive)**: `2,299` (Failure Rate: `99.74%`)
- **Mean Model A Probability**: `0.9750`
- **Mean Model B Probability**: `0.9753`
- **Mean Risk Difference ($\Delta P$)**: `+0.0004` if row.mean_risk_difference >= 0 else `0.0004`
- **Evidence Profile**: Parametric = `6.592`, Spatial = `0.672`, Block LRT = `-1.319`

---

## 3. Leakage Audit Checklist
- [x] **No Test Labels Used**: 0 competition test labels accessed.
- [x] **No Model Retraining**: Model weights & architecture untouched.
- [x] **No Labels Used in Clustering**: Ground truth labels used strictly post-hoc for descriptive summary statistics.
- [x] **Production Integrity**: `tuned/` and `submission.csv` strictly unmodified.

---

## 4. Conclusion & Scientific Interpretation
The multi-resolution evidence patterns successfully partition dies into distinct evidence-dominant signatures (Parametric-dominant, Spatial-dominant, Block-dominant, Multi-resolution, and Low-evidence baseline). These clusters provide valuable diagnostic groupings for yield engineers without implying causal failure mechanisms.
