# SanDisk Yield AI — Multi-Resolution Die Risk Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-frozen_production_release-brightgreen.svg)]()
[![Model B AP](https://img.shields.io/badge/Model_B_AP-0.6549-00F0FF.svg)]()
[![Tests](https://img.shields.io/badge/tests-104_passed-success.svg)]()
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](http://localhost:8501)

An end-to-end, interpretable multi-resolution die risk intelligence platform designed for **3D NAND flash memory wafer fabrication**. 

SanDisk Yield AI processes **154,037 eligible semiconductor dies** across **1,540 silicon wafers** to predict post-test die failures before final chip packaging, combining parametric electrical test measurements, spatial wafer geometry, and localized sub-die memory block scan readings into an additive log-odds evidence fusion architecture.

---

## 🔬 Key System Achievements

- **Champion Architecture (Model B)**: Combines a 500-feature parametric test encoder (`DiagonalScore`), pre-test spatial hazard density modeling, and a 73-feature Likelihood Ratio Test (LRT) sub-die memory block scan analyzer.
- **Primary Metric Lift**: Model B achieves an **Average Precision (AP) of 0.6549** compared to Baseline Model A's **0.5816**, delivering a **+12.59% relative AP gain** (+0.0733 absolute).
- **Secondary Metrics**: F1 score increases from **0.5505 to 0.6063** (+10.12%), Recall jumps from **42.40% to 51.16%** (+20.66% gain), and ROC-AUC improves from **0.8963 to 0.9255** (+3.26%).
- **Statistical Significance**: 1,000-wafer bootstrap 95% Confidence Interval for $\Delta\text{AP} = [+0.0677, +0.0783]$, Wilcoxon signed-rank test $p < 10^{-20}$, Cohen's effect size $d_z = 1.20$.
- **100% Zero Leakage**: Passed 13/13 leakage prevention audit checks. 5-Fold `StratifiedGroupKFold` strictly grouped by `wafer_id` ensures zero wafer overlap between training and validation folds.
- **Mathematical Interpretability**: Exact logit additive decomposition ($L_B = L_{\text{spatial}} + L_{\text{parametric}} + L_{\text{block}} - L_{\text{offset}}$) with zero reconstruction error ($\text{max residual} < 10^{-15}$).
- **Interactive Command Center**: 9-page Streamlit engineering analytics dashboard for real-time die risk inspection, logit waterfall attributions, sub-die block localization, failure signature clustering, and wafer yield excursion hotspots.

---

## 📊 Benchmark Results

Evaluated on **40 held-out test wafers** (39,351 dies, 32,598 eligible dies, 1,380 post-test failures):

| Model Architecture | Input Channels | AP | ROC-AUC | Failure F1 | Precision | Recall | AP Gain vs Baseline |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | 500 Parametric + Spatial Grid | 0.5816 | 0.8963 | 0.5505 | **0.7848** | 42.40% | Base Reference |
| **Model B (Production Champion)** | Parametric + Spatial + 73 Block LRT | **0.6549** | **0.9255** | **0.6063** | 0.7439 | **51.16%** | **+12.59% (+0.0733)** |

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 DIE INPUT DATA                                    |
|  500 Parametric Features | Pre-Test Spatial Grid | 2,000 Sub-Die Block Readings   |
+-----------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +-------------------+           +-------------------+
| PARAMETRIC BRANCH|           |  SPATIAL BRANCH   |           |   BLOCK BRANCH    |
|                  |           |                   |           |                   |
| 500 Features     |           | Spatial Kernel /  |           | 2,000 Readings    |
|        |         |           | Pre-Test Hazard   |           |        |          |
| Linear Discriminant|         | Density (w=3,5,7) |           | FFT Whitening     |
| Projection       |           | Radial & Edge     |           |        |          |
|        |         |           | Geometry          |           | Circular LRT Scan |
|        v         |           |        |          |           |        |          |
|  DiagonalScore   |           |        v          |           | 73 Block Features |
|   s_param        |           |  Log Hazard Rate  |           |        |          |
|        |         |           |    log(h_spatial) |           | Logistic Model    |
|        v         |           |        |          |           |        v          |
| Parametric Logit |           | Spatial Logit     |           | Block Logit       |
|    L_param       |           |    L_spatial      |           |    L_block        |
+------------------+           +-------------------+           +-------------------+
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        ADDITIVE MULTI-RESOLUTION FUSION                           |
|                                                                                   |
|        Final Logit: L_B = L_spatial + L_parametric + L_block - L_offset           |
|                                                                                   |
|                  Failure Probability: P_B = sigmoid(L_B)                          |
+-----------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +-------------------+           +-------------------+
| DIE EXPLANATION  |           | FAILURE SIGNATURES|           |  WAFER HOTSPOTS   |
| Logit Waterfall  |           | k=5 KMeans        |           | Connected-Component|
| Root Cause       |           | Evidence Archetypes|          | Yield Excursions  |
+------------------+           +-------------------+           +-------------------+
```

---

## 💡 Core Innovations

### 1. `DiagonalScore` Parametric Discriminant Encoder
Extracts the primary linear discriminant vector $\mathbf{w}$ across 500 parametric electrical measurements:
$$s_{\text{param}, i} = \sum_{j=1}^{500} w_j \cdot \frac{P_{i, j} - \mu_j}{\sigma_j}$$
The fitted discriminant vector recovers the true synthetic parametric failure degradation direction with **$r = 0.9985$ correlation**. Passes through a smooth B-spline logit transformation $L_{\text{parametric}} = f_{\text{spline}}(s_{\text{param}})$.

### 2. Circular Likelihood Ratio Test (LRT) Sub-Die Block Scan Encoder
Processes 2,000 correlated sub-die memory block readings per die:
1. **AR(1) Whitening**: Removes memory block spatial autocorrelation ($\rho \approx 0.82$).
2. **Circular Window Scanning**: Scans window sizes $W \in \{16, 32, 64, 128, 256, 384, 512\}$ to calculate Likelihood Ratio Test statistics $\Lambda(t, W)$.
3. **73-Feature Representation**: Combines multi-scale LRT peak scans, distribution quantiles, MAD, skewness, kurtosis, and autocorrelation.
4. **Logistic Block Logit**: Produces $L_{\text{block}} = \text{logit}(P_{\text{block\_raw}})$.

### 3. Additive Log-Odds Multi-Resolution Evidence Fusion
Combines independent evidence sources linearly in logit space:
$$L_{B, i} = L_{\text{spatial}, i} + L_{\text{parametric}, i} + L_{\text{block}, i} - 1.31$$
$$P_{B, i} = \frac{1}{1 + \exp(-L_{B, i})}$$
Decision rule threshold $\tau^* = 0.2912$ is derived strictly from 5-fold Out-Of-Fold (OOF) cross-validation.

---

## 📁 Repository Structure

```
sandisk_yield_ai/
├── demo/                               # Interactive Streamlit Analytics Dashboard
│   ├── app.py                          # Main Streamlit Application (9 Navigation Views)
│   ├── data_loader.py                  # Artifact Loader & Data Preparation Utilities
│   └── assets/                         # Dashboard Branding Assets
├── modeling/                           # Core Machine Learning & Validation Pipeline
│   ├── cache_features.py               # Dataset Caching & Feature Matrix Assembly
│   ├── features.py                     # Parametric & Spatial Feature Engineering
│   ├── run_experiments.py              # Cross-Validation Experiment Runner
│   └── validation.py                   # Metric Calculation & Wafer Fold Builders
├── mrf/                                # Sub-Die Block Scanning & LRT Processing
│   └── block.py                        # FFT Whitening & LRT Scan Feature Extractor
├── results/                            # Pre-computed Results, Models, & Figures
│   ├── final_validation/               # Champion Metrics & Final Submission Output
│   │   ├── final_metrics.json
│   │   ├── final_validation_report.md
│   │   └── submission.csv
│   └── figures/                        # Generated Diagnostic Plots
├── tests/                              # PyTest Automated Test Suite (104 Tests)
│   ├── test_adaptive_fusion.py
│   ├── test_blockcnn.py
│   ├── test_cache.py
│   ├── test_calibration.py
│   ├── test_demo.py
│   ├── test_experiments.py
│   ├── test_interpretability.py
│   ├── test_modeling.py
│   ├── test_mrf.py
│   ├── test_parametric_encoder.py
│   └── test_tuned.py
├── tuned/                              # Production Champion Pipeline & Modules
│   ├── adaptive_fusion.py
│   ├── blockcnn.py
│   ├── cache.py
│   ├── calibration.py
│   ├── experiment_final_validation.py  # Frozen Final Validation Runner
│   ├── hazard.py
│   ├── interpretability.py
│   ├── parametric_encoder.py
│   └── pipeline.py                     # Champion Model B Pipeline Architecture
├── config.yaml                         # Global Pipeline Configuration
├── generate_data.py                    # Synthetic Wafer Generation Script
├── README.md                           # Master Project Readme
├── RESULTS.md                          # Experiment Results Summary
├── RESULTS_TUNED.md                    # Tuned Champion Model Details
├── requirements.txt                    # Core Python Dependencies
└── submission.csv                      # Output Prediction CSV (39,351 rows)
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- `pip` or `uv` package manager

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/Crypto_Crafters/sandisk_yield_ai.git
cd sandisk_yield_ai

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Running the Pipeline & Dashboard

### 1. Launch Interactive Engineering Dashboard
Launch the Streamlit analytics app to explore predictions, logit evidence waterfalls, block localization, and wafer hotspots:
```bash
streamlit run demo/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Final Model B Validation Pipeline
To run the full frozen final validation suite and generate metric reports and submissions:
```bash
python tuned/experiment_final_validation.py
```

### 3. Run Synthetic Data Generator (Optional)
To generate raw synthetic 3D NAND wafer datasets from scratch:
```bash
python generate_data.py
```

### 4. Run Automated Test Suite
To execute the comprehensive PyTest suite covering all pipeline components:
```bash
python -m pytest tests/
```

---

## 🖥️ Interactive Dashboard Features

The Streamlit engineering dashboard (`demo/app.py`) provides 9 dedicated analytics views:

1. **🏠 Overview**: Executive summary, top KPI cards, model architecture flow, and all-wafer interactive grid explorer.
2. **🔍 Die Explanation**: Single-die deep dive with exact logit evidence waterfall breakdown and top parametric risk drivers.
3. **📊 Parametric Drivers**: Global feature importance rankings, discriminant coefficient distributions, and correlation maps.
4. **📈 Block Analysis**: Sub-die memory block anomaly localization, circular LRT scan profiles, and window size sensitivity.
5. **⚡ Model Benchmark**: Model A vs Model B comparative PR/ROC curves, threshold optimization, and metric tables.
6. **🎯 Case Studies**: Representative die case studies (High Risk Pass, Low Risk Failure, Spatial Edge Excursion, Block Anomaly).
7. **⚔️ Model A vs Model B**: Die-by-die prediction delta distribution ($\Delta P = P_B - P_A$) and discordant die analysis.
8. **🧩 Failure Signatures**: $k=5$ KMeans failure signature clustering of die evidence attributions.
9. **🔥 Wafer Hotspots**: Connected-component spatial cluster extraction for fab yield excursion detection.

---

## 🛡️ Target Rules & Zero Leakage Compliance

- **Pre-Test Defective Dies (`old_label == 1`)**: Completely excluded from training/validation eligibility. In final prediction output, pre-test failures are forced to `predicted_label = 1` by rule.
- **Target Isolation**: Ground truth targets (`label`) are strictly absent during feature creation, model fitting, and evidence attribution.
- **Wafer-Level Grouping**: All cross-validation splits use `StratifiedGroupKFold(n_splits=5)` grouped strictly by `wafer_id` to prevent intra-wafer data leakage.

---

## 📜 License & Citation

Developed for the **SanDisk Yield AI Hackathon Benchmark**. 
by Arunthathi R, Hemadharsini P, Abhinaya Rajesh
