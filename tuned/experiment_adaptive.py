"""Experiment 5: Adaptive / Gated Multi-Resolution Fusion Experiment.

Compares reference fixed Model B against an adaptive gated multi-resolution fusion model
across 5-fold wafer-grouped cross-validation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from tuned import pipeline
from tuned.adaptive_fusion import AdaptiveFusion
from tuned.cache import eligible_mask, load

RESULTS_DIR = Path("results/adaptive_fusion")


def evaluate_5fold_cv():
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

    # Baseline Model A metrics for reference
    model_a_ap = 0.582233
    model_a_auc = 0.896290
    model_a_f1 = 0.550088

    models_to_evaluate = [
        {
            "name": "model_b_fixed",
            "type": "fixed",
            "note": "Production Model B baseline (Fixed additive evidence fusion)",
        },
        {
            "name": "model_b_equal",
            "type": "equal",
            "note": "Equal-weighted baseline fusion (1/3 parametric, 1/3 spatial, 1/3 block)",
        },
        {
            "name": "model_b_adaptive",
            "type": "adaptive",
            "note": "Adaptive Gated Multi-Resolution Fusion (Learned linear softmax gate)",
        },
    ]

    all_summary = []
    all_fold_rows = []

    # Store OOF predictions and gate weights
    oof_preds = {m["name"]: np.zeros(len(eligible_indices), dtype=np.float64) for m in models_to_evaluate}
    oof_weights = np.zeros((len(eligible_indices), 3), dtype=np.float32)

    for model_info in models_to_evaluate:
        m_name = model_info["name"]
        m_type = model_info["type"]
        print(f"\n--- Evaluating {m_name}: {model_info['note']} ---", flush=True)

        t0 = time.time()
        fold_aps, fold_aucs, fold_f1s = [], [], []

        for fold_idx, (train_local_idx, val_local_idx) in enumerate(splits, start=1):
            train_global_idx = eligible_indices[train_local_idx]
            val_global_idx = eligible_indices[val_local_idx]

            # Fit rows mask for training
            train_rows = np.zeros(len(frame), dtype=bool)
            train_rows[train_global_idx] = True

            # Val rows mask for inference
            val_rows = np.zeros(len(frame), dtype=bool)
            val_rows[val_global_idx] = True

            y_val = label[val_global_idx]

            if m_type == "fixed":
                fusion = pipeline.Fusion(
                    use_block=True, use_parametric=True, use_hazard=True, correct_prior=True, random_state=42
                )
                fusion.fit(frame, x, train_rows)
                val_probs = fusion.predict_proba(frame, x, val_rows)

            elif m_type == "equal":
                # Equal weight fusion
                fusion = pipeline.Fusion(
                    use_block=True, use_parametric=True, use_hazard=True, correct_prior=True, random_state=42
                )
                fusion.fit(frame, x, train_rows)
                scored = fusion.score_frame(frame, x, val_rows)
                e_p = scored["parametric_score"].to_numpy()
                e_b = scored["block_score"].to_numpy()
                h = scored["hazard"].to_numpy()
                w_ids = frame.loc[val_rows, "wafer_id"].astype(str).to_numpy()
                w_rates = np.array([fusion.fitted_rates_.get(w, fusion.overall_rate_) for w in w_ids])
                e_s = pipeline._logit(np.clip(w_rates * h, 1e-9, pipeline.CAP))
                
                # Equal weight (1/3 each) multiplied by 3
                equal_log_odds = e_p + e_s + e_b
                val_probs = 1.0 / (1.0 + np.exp(-equal_log_odds))

            elif m_type == "adaptive":
                base_fusion = pipeline.Fusion(
                    use_block=True, use_parametric=True, use_hazard=True, correct_prior=True, random_state=42
                )
                adaptive = AdaptiveFusion(base_fusion=base_fusion, gate_epochs=150, lr=0.02, random_state=42)
                adaptive.fit(frame, x, train_rows)
                val_probs, val_w = adaptive.predict_proba(frame, x, val_rows)
                oof_weights[val_local_idx] = val_w

            oof_preds[m_name][val_local_idx] = val_probs

            f_ap = average_precision_score(y_val, val_probs)
            f_auc = roc_auc_score(y_val, val_probs)
            
            # Simple threshold optimization for fold metric
            best_thresh = 0.35
            best_f1 = 0.0
            for th in np.linspace(0.1, 0.6, 51):
                f1_cand = f1_score(y_val, (val_probs >= th).astype(int), zero_division=0)
                if f1_cand > best_f1:
                    best_f1 = f1_cand
                    best_thresh = th

            fold_aps.append(f_ap)
            fold_aucs.append(f_auc)
            fold_f1s.append(best_f1)

            all_fold_rows.append({
                "experiment": m_name,
                "fold": fold_idx,
                "ap": f_ap,
                "roc_auc": f_auc,
                "f1": best_f1,
                "threshold": best_thresh,
                "val_dies": len(val_local_idx),
                "val_failures": int(y_val.sum()),
            })

            print(f"  Fold {fold_idx}/5 - AP: {f_ap:.4f}, ROC-AUC: {f_auc:.4f}, F1: {best_f1:.4f}", flush=True)

        elapsed = time.time() - t0

        # Overall OOF evaluation
        oof_p = oof_preds[m_name]
        overall_ap = average_precision_score(y_eligible, oof_p)
        overall_auc = roc_auc_score(y_eligible, oof_p)

        # Optimize overall threshold
        best_opt_thresh = 0.35
        best_overall_f1 = 0.0
        best_prec, best_rec = 0.0, 0.0
        for th in np.linspace(0.05, 0.70, 131):
            preds_binary = (oof_p >= th).astype(int)
            f1_cand = f1_score(y_eligible, preds_binary, zero_division=0)
            if f1_cand > best_overall_f1:
                best_overall_f1 = f1_cand
                best_opt_thresh = th
                best_prec = precision_score(y_eligible, preds_binary, zero_division=0)
                best_rec = recall_score(y_eligible, preds_binary, zero_division=0)

        fixed_ap = all_summary[0]["average_precision"] if len(all_summary) > 0 else overall_ap

        summary_item = {
            "experiment": m_name,
            "note": model_info["note"],
            "average_precision": overall_ap,
            "roc_auc": overall_auc,
            "f1": best_overall_f1,
            "precision": best_prec,
            "recall": best_rec,
            "opt_threshold": float(best_opt_thresh),
            "ap_mean": float(np.mean(fold_aps)),
            "ap_std": float(np.std(fold_aps)),
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
            "f1_mean": float(np.mean(fold_f1s)),
            "f1_std": float(np.std(fold_f1s)),
            "seconds": round(elapsed, 1),
            "delta_ap_vs_fixed_b": overall_ap - fixed_ap,
            "delta_ap_vs_model_a": overall_ap - model_a_ap,
            "delta_auc_vs_fixed_b": overall_auc - (all_summary[0]["roc_auc"] if len(all_summary) > 0 else overall_auc),
            "delta_f1_vs_fixed_b": best_overall_f1 - (all_summary[0]["f1"] if len(all_summary) > 0 else best_overall_f1),
        }
        all_summary.append(summary_item)

        print(f"Summary {m_name}: AP={overall_ap:.6f} (std={np.std(fold_aps):.4f}), ROC-AUC={overall_auc:.6f}, F1={best_overall_f1:.6f} ({elapsed:.1f}s)", flush=True)

    # Save summary CSV and JSON
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)
    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)

    # 4. Compute Gate Statistics & Interpretability Samples
    w_param = oof_weights[:, 0]
    w_spatial = oof_weights[:, 1]
    w_block = oof_weights[:, 2]

    gate_stats = {
        "overall_mean_weights": {
            "w_parametric": float(np.mean(w_param)),
            "w_spatial": float(np.mean(w_spatial)),
            "w_block": float(np.mean(w_block)),
        },
        "overall_std_weights": {
            "w_parametric": float(np.std(w_param)),
            "w_spatial": float(np.std(w_spatial)),
            "w_block": float(np.std(w_block)),
        },
        "w_parametric_distribution": {
            "min": float(np.min(w_param)),
            "p25": float(np.percentile(w_param, 25)),
            "p50": float(np.median(w_param)),
            "p75": float(np.percentile(w_param, 75)),
            "max": float(np.max(w_param)),
        },
        "w_spatial_distribution": {
            "min": float(np.min(w_spatial)),
            "p25": float(np.percentile(w_spatial, 25)),
            "p50": float(np.median(w_spatial)),
            "p75": float(np.percentile(w_spatial, 75)),
            "max": float(np.max(w_spatial)),
        },
        "w_block_distribution": {
            "min": float(np.min(w_block)),
            "p25": float(np.percentile(w_block, 25)),
            "p50": float(np.median(w_block)),
            "p75": float(np.percentile(w_block, 75)),
            "max": float(np.max(w_block)),
        },
    }

    with open(RESULTS_DIR / "gate_statistics.json", "w") as f:
        json.dump(gate_stats, f, indent=2)

    # Sample dies representing different dominant evidence channels
    eligible_df = frame.iloc[eligible_indices].copy().reset_index(drop=True)
    eligible_df["w_param"] = w_param
    eligible_df["w_spatial"] = w_spatial
    eligible_df["w_block"] = w_block
    eligible_df["pred_prob_adaptive"] = oof_preds["model_b_adaptive"]
    eligible_df["pred_prob_fixed"] = oof_preds["model_b_fixed"]

    samples = pd.concat([
        eligible_df.sort_values("w_param", ascending=False).head(5).assign(dominant_channel="parametric"),
        eligible_df.sort_values("w_spatial", ascending=False).head(5).assign(dominant_channel="spatial"),
        eligible_df.sort_values("w_block", ascending=False).head(5).assign(dominant_channel="block"),
    ])

    sample_cols = [c for c in ["wafer_id", "die_x", "die_y", "label", "dominant_channel", "w_param", "w_spatial", "w_block", "pred_prob_fixed", "pred_prob_adaptive"] if c in eligible_df.columns]
    samples[sample_cols].to_csv(RESULTS_DIR / "gate_weight_samples.csv", index=False)

    # Write Markdown Experiment Notes
    fixed_res = all_summary[0]
    adapt_res = all_summary[2]
    delta_ap = adapt_res["average_precision"] - fixed_res["average_precision"]
    delta_auc = adapt_res["roc_auc"] - fixed_res["roc_auc"]
    delta_f1 = adapt_res["f1"] - fixed_res["f1"]

    decision = "KEEP" if (delta_ap >= 0.005 and adapt_res["ap_std"] <= fixed_res["ap_std"] * 1.2) else "REJECT"

    md_content = f"""# Experiment 5: Adaptive / Gated Multi-Resolution Fusion

## Executive Summary
This experiment tests whether a small, interpretable 4-feature linear softmax gating model (`AdaptiveFusion`) that dynamically balances Parametric, Spatial, and Block LRT evidence can improve upon the current fixed Model B fusion.

## Benchmark Results (5-Fold Wafer-Grouped CV)
- **Model A Baseline**: AP = {model_a_ap:.6f}, ROC-AUC = {model_a_auc:.6f}, F1 = {model_a_f1:.6f}
- **Model B Fixed Baseline**: AP = {fixed_res['average_precision']:.6f}, ROC-AUC = {fixed_res['roc_auc']:.6f}, F1 = {fixed_res['f1']:.6f}
- **Model B Adaptive Gated**: AP = {adapt_res['average_precision']:.6f}, ROC-AUC = {adapt_res['roc_auc']:.6f}, F1 = {adapt_res['f1']:.6f}

### Summary Table
| Experiment | AP | ROC-AUC | F1 | Precision | Recall | Δ AP vs Fixed B | Δ AP vs Model A | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| {fixed_res['experiment']} | {fixed_res['average_precision']:.6f} | {fixed_res['roc_auc']:.6f} | {fixed_res['f1']:.6f} | {fixed_res['precision']:.4f} | {fixed_res['recall']:.4f} | +0.000000 | +{fixed_res['delta_ap_vs_model_a']:.6f} | {fixed_res['seconds']}s |
| {all_summary[1]['experiment']} | {all_summary[1]['average_precision']:.6f} | {all_summary[1]['roc_auc']:.6f} | {all_summary[1]['f1']:.6f} | {all_summary[1]['precision']:.4f} | {all_summary[1]['recall']:.4f} | {all_summary[1]['delta_ap_vs_fixed_b']:.6f} | +{all_summary[1]['delta_ap_vs_model_a']:.6f} | {all_summary[1]['seconds']}s |
| {adapt_res['experiment']} | {adapt_res['average_precision']:.6f} | {adapt_res['roc_auc']:.6f} | {adapt_res['f1']:.6f} | {adapt_res['precision']:.4f} | {adapt_res['recall']:.4f} | {delta_ap:+.6f} | +{adapt_res['delta_ap_vs_model_a']:.6f} | {adapt_res['seconds']}s |

## Key Metric Deltas (Adaptive vs Fixed Model B)
- **Δ AP**: {delta_ap:+.6f}
- **Δ ROC-AUC**: {delta_auc:+.6f}
- **Δ F1 Score**: {delta_f1:+.6f}

## Gate Weight Statistics
- **Average Weights**: Parametric = {gate_stats['overall_mean_weights']['w_parametric']:.4f}, Spatial = {gate_stats['overall_mean_weights']['w_spatial']:.4f}, Block = {gate_stats['overall_mean_weights']['w_block']:.4f}
- **Distribution Range**:
  - Parametric: [{gate_stats['w_parametric_distribution']['min']:.4f}, {gate_stats['w_parametric_distribution']['max']:.4f}]
  - Spatial: [{gate_stats['w_spatial_distribution']['min']:.4f}, {gate_stats['w_spatial_distribution']['max']:.4f}]
  - Block: [{gate_stats['w_block_distribution']['min']:.4f}, {gate_stats['w_block_distribution']['max']:.4f}]

## Final Decision
**EXPERIMENT 5 STATUS = {decision}**

### Decision Rationale
- Fixed Model B AP: {fixed_res['average_precision']:.6f}
- Adaptive Model B AP: {adapt_res['average_precision']:.6f}
- Delta AP: {delta_ap:+.6f}
"""

    with open(RESULTS_DIR / "experiment_notes.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nExperiment 5 complete! Results written to {RESULTS_DIR}")
    print(f"EXPERIMENT 5 STATUS = {decision}")


if __name__ == "__main__":
    evaluate_5fold_cv()
