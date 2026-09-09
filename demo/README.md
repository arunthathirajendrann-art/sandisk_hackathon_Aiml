# SanDisk Yield AI — Multi-Resolution Die Risk Explorer

The **Multi-Resolution Die Risk Explorer** is a lightweight, interactive local web application built using **Streamlit** and **Plotly**. It visualizes predictions, logit evidence attributions, sub-die memory block sequence anomaly localizations, and benchmark comparisons for production **Model B**.

---

## 1. Quick Start Guide

### Prerequisites
Make sure `streamlit`, `plotly`, and `pandas` are installed in your Python environment:
```bash
pip install streamlit plotly pandas numpy
```

### Launch Command
To launch the interactive dashboard locally, execute:
```bash
streamlit run demo/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 2. Data Sources Used
The dashboard relies **strictly on pre-computed interpretability and validation artifacts** without running model inference or modifying ML code:
- `results/interpretability/explanation_samples.json`: Structured die explanations and top 10 parametric measurement drivers.
- `results/interpretability/wafer_heatmap_data.csv`: Spatial die grid coordinates $(r, c)$ and predicted probabilities $P(\text{fail})$.
- `results/interpretability/block_anomaly_samples.csv`: Localized sub-die block scan anomaly peak indices $t^* \in [0, 1999]$.
- `results/interpretability/contribution_summary.csv`: Additive logit evidence breakdown ($C_{\text{spatial}}, C_{\text{param}}, C_{\text{block}}$).
- `results/final_validation/final_metrics.json`: Final Model A vs. Model B benchmark metrics.
- `results/interpretability/plots/`: High-resolution wafer heatmaps and sub-die block sequence profiles.

---

## 3. Dashboard Sections Breakdown

1. **🏠 Overview (Wafer Spatial Overview)**
   - Dynamic wafer selector covering all test wafers.
   - Interactive Plotly scatter heatmap of wafer die coordinates $(r, c)$ colored by predicted failure probability $P(\text{fail})$.
   - Interactive die row/col selectors & die intelligence panel.

2. **🔍 Die Explanation (Logit Evidence Waterfall)**
   - Metric cards: Failure probability $P(\text{fail})$, Final Logit, Risk Classification (`CRITICAL RISK`, `ELEVATED RISK`, `PASS`).
   - Additive evidence waterfall flow chart: $C_{\text{spatial}} + C_{\text{param}} + C_{\text{block}} \approx \text{final\_logit}$.
   - Verification badge verifying exact logit identity sum checks.

3. **📊 Parametric Drivers**
   - Top 10 positive (risk-increasing) and top 10 negative (risk-reducing) measurement feature drivers for selected die.
   - Color-coded bar chart of feature logit contributions.

4. **📈 Block Analysis (Sub-Die Block Anomaly Localization)**
   - Raw 2,000 sub-die memory block sequence readings vs position $0 \dots 1999$.
   - Overlays circular FFT-whitened LRT scan statistic highlighting exact peak anomaly index $t^*$.

5. **⚡ Model Benchmark**
   - Side-by-side performance comparison of Baseline Model A against Production Model B Champion.
   - Highlights +12.59% relative AP gain (0.5816 $\rightarrow$ 0.6549) and +10.12% relative F1 gain (0.5505 $\rightarrow$ 0.6063).

6. **🎯 Case Studies**
   - Pre-configured shortcuts for True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) case studies.

7. **⚔️ Model A vs Model B (Wafer Comparison)**
   - Three-panel interactive wafer map: Model A Risk | Model B Risk | Risk Difference ($B - A$).
   - Direct die-level comparison across models on identical coordinates.

8. **🧩 Failure Signatures (Step 5.5 Interactive Drill-Down)**
   - Discovered multi-resolution evidence signatures (Parametric-dominant, Block-dominant, Spatial-dominant, Low-evidence baseline).
   - Interactive drill-down flow: Signature $\rightarrow$ Affected Wafers $\rightarrow$ Affected Dies $\rightarrow$ Die Explanation.

9. **🔥 Wafer Hotspots (Step 6 High-Risk Region Explorer)**
   - 8-neighbor spatial connected-component hotspot detection on Model B's risk surface.
   - Visual bounding box highlighting, centroid annotations, isolated high-risk die markings, and direct die explanation drill-down.

---

## 4. Production Champion Architecture & Additive Evidence
- **Model A (Baseline)**: `DiagonalScore` parametric evidence + spatial hazard evidence.
- **Model B (Champion)**: `DiagonalScore` + spatial hazard + 73-feature LRT block encoder + additive log-odds fusion.
- **Fixed Additive Log-Odds Fusion**:
  $$\text{final\_logit} = C_{\text{spatial}} + C_{\text{param}} + C_{\text{block}}$$
- **Validation**: 5-Fold `StratifiedGroupKFold` grouped by `wafer_id`.

---

## 5. Interpretability Disclaimer
> ℹ️ **Observational Interpretability Note**: Evidence contributions describe how trained Model B formed its prediction log-odds. They represent exact observational model attributions, not causal physical defect proof.
