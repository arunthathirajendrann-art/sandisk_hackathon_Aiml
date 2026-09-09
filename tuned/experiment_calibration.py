"""Experiment 6: Probability Calibration Experiment.

Evaluates raw Model B against Platt scaling (sigmoid calibration) and Isotonic regression
across 5-fold wafer-grouped cross-validation using strict internal out-of-fold calibration fitting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from tuned.cache import eligible_mask, load
from tuned.calibration import IsotonicCalibrator, PlattCalibrator, compute_calibration_curve
from tuned.pipeline import Fusion

RESULTS_DIR = Path("results/calibration")


def evaluate_calibration_cv():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from input/cache_train...", flush=True)
    frame = load(Path("input/cache_train"))
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)

    label = frame["label"].to_numpy(dtype=np.int8)
    old_label = frame["old_label"].to_numpy(dtype=np.int8)
    wafer = frame["wafer_id"].astype(str).to_numpy()

    # Eligible population mask
    eligible_mask = (old_label == 0)
    eligible_indices = np.flatnonzero(eligible_mask)
    y_eligible = label[eligible_indices]
    wafer_eligible = wafer[eligible_indices]

    print(f"Dataset loaded: {len(frame)} total dies, {len(eligible_indices)} eligible dies ({y_eligible.sum()} failures, {y_eligible.mean():.4%}).", flush=True)

    # 5-fold StratifiedGroupKFold on wafer_id
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(eligible_indices, y_eligible, groups=wafer_eligible))

    methods = ["model_b_raw", "model_b_platt", "model_b_isotonic"]
    oof_preds = {m: np.zeros(len(eligible_indices), dtype=np.float64) for m in methods}
    fold_records = []

    t_start = time.time()

    for fold_idx, (train_local_idx, val_local_idx) in enumerate(splits, start=1):
        print(f"\n--- Outer Fold {fold_idx}/5 ---", flush=True)

        train_global_idx = eligible_indices[train_local_idx]
        val_global_idx = eligible_indices[val_local_idx]

        train_rows = np.zeros(len(frame), dtype=bool)
        train_rows[train_global_idx] = True

        val_rows = np.zeros(len(frame), dtype=bool)
        val_rows[val_global_idx] = True

        y_train = label[train_global_idx]
        y_val = label[val_global_idx]
        wafer_train = wafer[train_global_idx]

        # 1. Fit outer Model B on outer training fold
        outer_model = Fusion(use_block=True, use_parametric=True, use_hazard=True, correct_prior=True, random_state=42)
        outer_model.fit(frame, x, train_rows)
        raw_val_probs = outer_model.predict_proba(frame, x, val_rows)

        # 2. Internal 5-fold CV on training fold to generate internal OOF raw probabilities
        internal_sgkf = StratifiedGroupKFold(n_splits=5)
        internal_splits = list(internal_sgkf.split(train_local_idx, y_train, groups=wafer_train))
        internal_train_oof_probs = np.zeros(len(train_local_idx), dtype=np.float64)

        for int_train_idx, int_val_idx in internal_splits:
            int_train_global = train_global_idx[int_train_idx]
            int_val_global = train_global_idx[int_val_idx]

            int_train_rows = np.zeros(len(frame), dtype=bool)
            int_train_rows[int_train_global] = True

            int_val_rows = np.zeros(len(frame), dtype=bool)
            int_val_rows[int_val_global] = True

            int_model = Fusion(use_block=True, use_parametric=True, use_hazard=True, correct_prior=True, random_state=42)
            int_model.fit(frame, x, int_train_rows)
            internal_train_oof_probs[int_val_idx] = int_model.predict_proba(frame, x, int_val_rows)

        # 3. Fit Platt and Isotonic calibrators on internal OOF probabilities
        platt_cal = PlattCalibrator(random_state=42).fit(internal_train_oof_probs, y_train)
        iso_cal = IsotonicCalibrator().fit(internal_train_oof_probs, y_train)

        # 4. Apply calibrators to outer validation predictions
        platt_val_probs = platt_cal.predict_proba(raw_val_probs)
        iso_val_probs = iso_cal.predict_proba(raw_val_probs)

        oof_preds["model_b_raw"][val_local_idx] = raw_val_probs
        oof_preds["model_b_platt"][val_local_idx] = platt_val_probs
        oof_preds["model_b_isotonic"][val_local_idx] = iso_val_probs

        for m_name, probs in [
            ("model_b_raw", raw_val_probs),
            ("model_b_platt", platt_val_probs),
            ("model_b_isotonic", iso_val_probs),
        ]:
            ap = average_precision_score(y_val, probs)
            auc = roc_auc_score(y_val, probs)
            brier = brier_score_loss(y_val, probs)
            logloss = log_loss(y_val, np.clip(probs, 1e-15, 1.0 - 1e-15))
            
            # Best threshold for fold F1
            best_th, best_f1 = 0.35, 0.0
            for th in np.linspace(0.1, 0.6, 51):
                f1_cand = f1_score(y_val, (probs >= th).astype(int), zero_division=0)
                if f1_cand > best_f1:
                    best_f1, best_th = f1_cand, th

            fold_records.append({
                "experiment": m_name,
                "fold": fold_idx,
                "ap": ap,
                "roc_auc": auc,
                "f1": best_f1,
                "brier_score": brier,
                "log_loss": logloss,
                "threshold": best_th,
            })

            print(f"  {m_name:<16} - AP: {ap:.4f}, ROC-AUC: {auc:.4f}, F1: {best_f1:.4f}, Brier: {brier:.6f}, LogLoss: {logloss:.6f}", flush=True)

    elapsed_time = time.time() - t_start

    # Aggregate summaries across all 5 folds
    summaries = []
    calibration_curves = []

    raw_summary = None

    for m_name in methods:
        p_oof = oof_preds[m_name]
        ap_overall = average_precision_score(y_eligible, p_oof)
        auc_overall = roc_auc_score(y_eligible, p_oof)
        brier_overall = brier_score_loss(y_eligible, p_oof)
        logloss_overall = log_loss(y_eligible, np.clip(p_oof, 1e-15, 1.0 - 1e-15))

        # Calibration curve and ECE
        df_curve, ece = compute_calibration_curve(y_eligible, p_oof, n_bins=10)
        df_curve["experiment"] = m_name
        calibration_curves.append(df_curve)

        # Optimize overall threshold
        best_th, best_f1, best_prec, best_rec = 0.35, 0.0, 0.0, 0.0
        for th in np.linspace(0.05, 0.70, 131):
            p_bin = (p_oof >= th).astype(int)
            f1_c = f1_score(y_eligible, p_bin, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_th = th
                best_prec = precision_score(y_eligible, p_bin, zero_division=0)
                best_rec = recall_score(y_eligible, p_bin, zero_division=0)

        m_folds = [r for r in fold_records if r["experiment"] == m_name]

        item = {
            "experiment": m_name,
            "average_precision": ap_overall,
            "roc_auc": auc_overall,
            "f1": best_f1,
            "precision": best_prec,
            "recall": best_rec,
            "brier_score": brier_overall,
            "log_loss": logloss_overall,
            "ece": ece,
            "opt_threshold": float(best_th),
            "ap_mean": float(np.mean([r["ap"] for r in m_folds])),
            "ap_std": float(np.std([r["ap"] for r in m_folds])),
            "auc_mean": float(np.mean([r["roc_auc"] for r in m_folds])),
            "auc_std": float(np.std([r["roc_auc"] for r in m_folds])),
            "f1_mean": float(np.mean([r["f1"] for r in m_folds])),
            "f1_std": float(np.std([r["f1"] for r in m_folds])),
            "brier_mean": float(np.mean([r["brier_score"] for r in m_folds])),
            "brier_std": float(np.std([r["brier_score"] for r in m_folds])),
            "logloss_mean": float(np.mean([r["log_loss"] for r in m_folds])),
            "logloss_std": float(np.std([r["log_loss"] for r in m_folds])),
            "seconds": round(elapsed_time / 3, 1),
        }

        if m_name == "model_b_raw":
            raw_summary = item
            item["delta_brier"] = 0.0
            item["delta_logloss"] = 0.0
            item["delta_ap"] = 0.0
            item["delta_roc_auc"] = 0.0
            item["delta_f1"] = 0.0
            item["status"] = "BASELINE"
        else:
            item["delta_brier"] = brier_overall - raw_summary["brier_score"]
            item["delta_logloss"] = logloss_overall - raw_summary["log_loss"]
            item["delta_ap"] = ap_overall - raw_summary["average_precision"]
            item["delta_roc_auc"] = auc_overall - raw_summary["roc_auc"]
            item["delta_f1"] = best_f1 - raw_summary["f1"]
            
            # KEEP decision rule: materially improves Brier/LogLoss/ECE without damaging AP
            if (item["delta_brier"] <= -0.0001 or item["delta_logloss"] <= -0.001) and item["delta_ap"] >= -0.005:
                item["status"] = "KEEP"
            else:
                item["status"] = "REJECT"

        summaries.append(item)

    # Save summary CSV & JSON
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)
    with open(RESULTS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    pd.DataFrame(fold_records).to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.concat(calibration_curves, ignore_index=True).to_csv(RESULTS_DIR / "calibration_curve.csv", index=False)

    # Generate Markdown notes
    platt_item = [s for s in summaries if s["experiment"] == "model_b_platt"][0]
    iso_item = [s for s in summaries if s["experiment"] == "model_b_isotonic"][0]

    overall_status = "KEEP" if (platt_item["status"] == "KEEP" or iso_item["status"] == "KEEP") else "REJECT"

    md_notes = f"""# Experiment 6: Probability Calibration Evaluation

## Executive Summary
This experiment evaluates whether Platt scaling (sigmoid calibration) or Isotonic regression improves the probability reliability (Brier Score, Log Loss, ECE) of production Model B without degrading ranking performance (AP, ROC-AUC).

## Benchmark Results (5-Fold Wafer-Grouped CV)
| Model | AP | ROC-AUC | F1 | Brier Score | Log Loss | ECE | Δ Brier | Δ LogLoss | Δ AP | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| model_b_raw | {raw_summary['average_precision']:.6f} | {raw_summary['roc_auc']:.6f} | {raw_summary['f1']:.6f} | {raw_summary['brier_score']:.6f} | {raw_summary['log_loss']:.6f} | {raw_summary['ece']:.6f} | +0.000000 | +0.000000 | +0.000000 | BASELINE |
| model_b_platt | {platt_item['average_precision']:.6f} | {platt_item['roc_auc']:.6f} | {platt_item['f1']:.6f} | {platt_item['brier_score']:.6f} | {platt_item['log_loss']:.6f} | {platt_item['ece']:.6f} | {platt_item['delta_brier']:+.6f} | {platt_item['delta_logloss']:+.6f} | {platt_item['delta_ap']:+.6f} | **{platt_item['status']}** |
| model_b_isotonic | {iso_item['average_precision']:.6f} | {iso_item['roc_auc']:.6f} | {iso_item['f1']:.6f} | {iso_item['brier_score']:.6f} | {iso_item['log_loss']:.6f} | {iso_item['ece']:.6f} | {iso_item['delta_brier']:+.6f} | {iso_item['delta_logloss']:+.6f} | {iso_item['delta_ap']:+.6f} | **{iso_item['status']}** |

## Calibration Curve Summary (10 Bins)
- **Raw Model B ECE**: {raw_summary['ece']:.6f}
- **Platt Model B ECE**: {platt_item['ece']:.6f}
- **Isotonic Model B ECE**: {iso_item['ece']:.6f}

## Final Decision
**EXPERIMENT 6 STATUS = {overall_status}**

### Key Observations
- Platt Scaling Decision: **{platt_item['status']}**
- Isotonic Regression Decision: **{iso_item['status']}**
- **Production Integration Note**: Production Model B remains 100% untouched.
"""

    with open(RESULTS_DIR / "experiment_notes.md", "w", encoding="utf-8") as f:
        f.write(md_notes)

    print(f"\nExperiment 6 complete! Results written to {RESULTS_DIR}", flush=True)
    print(f"EXPERIMENT 6 STATUS = {overall_status}", flush=True)


if __name__ == "__main__":
    evaluate_calibration_cv()
