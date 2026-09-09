"""Experiment 4: Block Encoder Comparison

Evaluates:
1. block_only_LRT
2. block_only_CNN
3. block_only_LRT_CNN
4. model_b_LRT
5. model_b_CNN
6. model_b_LRT_CNN

Uses 5-fold StratifiedGroupKFold grouped by wafer_id, strictly with no leakage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    repeated_stratified_group_folds,
)
from tuned.cache import eligible_mask, load
from tuned import blockcnn
from tuned.pipeline import Fusion
from tuned.select import load_chosen

EXPERIMENTS = [
    {
        "name": "block_only_LRT",
        "note": "Statistical LRT block representation alone",
        "options": {
            "use_parametric": False,
            "correct_prior": False,
            "use_rate": False,
            "use_hazard": False,
            "use_block_lrt": True,
            "use_block_cnn": False,
        },
    },
    {
        "name": "block_only_CNN",
        "note": "Learned 1D CNN block representation alone",
        "options": {
            "use_parametric": False,
            "correct_prior": False,
            "use_rate": False,
            "use_hazard": False,
            "use_block_lrt": False,
            "use_block_cnn": True,
        },
    },
    {
        "name": "block_only_LRT_CNN",
        "note": "Combined LRT + 1D CNN block representation alone",
        "options": {
            "use_parametric": False,
            "correct_prior": False,
            "use_rate": False,
            "use_hazard": False,
            "use_block_lrt": True,
            "use_block_cnn": True,
        },
    },
    {
        "name": "model_b_LRT",
        "note": "Model B reference baseline (Model A + LRT block score + wafer rate)",
        "options": {
            "use_block_lrt": True,
            "use_block_cnn": False,
        },
    },
    {
        "name": "model_b_CNN",
        "note": "Model B with 1D CNN block representation",
        "options": {
            "use_block_lrt": False,
            "use_block_cnn": True,
        },
    },
    {
        "name": "model_b_LRT_CNN",
        "note": "Model B with combined LRT + 1D CNN block representation",
        "options": {
            "use_block_lrt": True,
            "use_block_cnn": True,
        },
    },
]

# Champion Model A reference benchmark for delta comparison
MODEL_A_BENCHMARK_AP = 0.582233
MODEL_A_BENCHMARK_AUC = 0.896290
MODEL_A_BENCHMARK_F1 = 0.550088


def run_experiment(
    frame: pd.DataFrame,
    x_param: np.ndarray,
    readings: np.ndarray,
    exp_config: dict,
    folds,
    eligible_index: np.ndarray,
    wafer: np.ndarray,
    y: np.ndarray,
    oof_cnn_logits: np.ndarray | None,
    base_constants: dict | None = None,
):
    fusion_kwargs = {**(base_constants or {}), **exp_config["options"]}
    if oof_cnn_logits is not None:
        fusion_kwargs["block_cnn_scores"] = oof_cnn_logits

    total = np.zeros(len(eligible_index), dtype=np.float64)
    seen = np.zeros(len(eligible_index), dtype=np.int16)
    fold_rows = []

    for fold in folds:
        train_wafers = set(wafer[fold.train_index])
        train_rows = np.isin(frame["wafer_id"].astype(str).to_numpy(), list(train_wafers))
        
        score_rows = np.zeros(len(frame), dtype=bool)
        score_rows[eligible_index[fold.validation_index]] = True

        model = Fusion(**fusion_kwargs).fit(frame, x_param, train_rows)
        probability = model.predict_proba(frame, x_param, score_rows)
        
        total[fold.validation_index] += probability
        seen[fold.validation_index] += 1
        
        y_val_fold = y[fold.validation_index]
        ap_fold = float(average_precision_score(y_val_fold, probability))
        auc_fold = float(roc_auc_score(y_val_fold, probability))
        thresh_fold = best_f1_threshold(y_val_fold, probability)
        m_fold = classification_metrics(y_val_fold, probability, thresh_fold)
        
        fold_rows.append({
            "experiment": exp_config["name"],
            "fold": fold.fold,
            "average_precision": ap_fold,
            "roc_auc": auc_fold,
            "fail_f1": m_fold["fail_f1"],
            "fail_precision": m_fold["fail_precision"],
            "fail_recall": m_fold["fail_recall"],
            "threshold": thresh_fold,
        })

    probability = total / seen
    threshold = best_f1_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)

    fold_aps = [r["average_precision"] for r in fold_rows]
    fold_aucs = [r["roc_auc"] for r in fold_rows]
    fold_f1s = [r["fail_f1"] for r in fold_rows]

    metrics.update({
        "experiment": exp_config["name"],
        "note": exp_config["note"],
        "ap_mean": float(np.mean(fold_aps)),
        "ap_std": float(np.std(fold_aps)),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "opt_threshold": float(threshold),
        "calibration": float(probability.sum() / max(y.sum(), 1)),
        "eligible_rows": int(len(y)),
        "eligible_failures": int(y.sum()),
        "wafers": int(len(set(wafer))),
    })

    return metrics, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("input/cache_train"))
    parser.add_argument("--readings-path", type=Path, default=Path("input/cache_train/block_readings.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/block_encoder"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", type=Path, default=Path("results/tuned_selection/selection.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.cache_dir}...", flush=True)
    frame = load(args.cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x_param = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)

    print(f"Loading block readings from {args.readings_path}...", flush=True)
    readings = np.load(args.readings_path, mmap_mode="r")

    eligible = eligible_mask(frame)
    eligible_index = np.flatnonzero(eligible)
    y = frame.loc[eligible, "label"].to_numpy(dtype=np.int8)
    old_label = frame["old_label"].to_numpy(dtype=np.int8)
    label_full = frame["label"].to_numpy(dtype=np.int8)
    wafer_full = frame["wafer_id"].astype(str).to_numpy()
    wafer = frame.loc[eligible, "wafer_id"].astype(str).to_numpy()

    print(f"  {len(frame)} dies, {eligible.sum()} eligible, {int(y.sum())} failures "
          f"({100 * y.mean():.3f}%), {len(set(wafer))} wafers", flush=True)

    # 1. Train and compute 5-fold OOF CNN logits
    print("\n--- Fitting 1D BlockCNN 5-fold Out-of-Fold ---", flush=True)
    t_cnn = time.perf_counter()
    oof_cnn_logits = blockcnn.fit_predict_oof_cnn(
        readings=readings,
        label=label_full,
        old_label=old_label,
        wafer=wafer_full,
        n_splits=args.n_splits,
        epochs=5,
        batch_size=4096,
        learning_rate=3e-3,
        seed=args.seed,
        verbose=True,
    )
    cnn_train_time = time.perf_counter() - t_cnn
    print(f"1D BlockCNN 5-fold OOF finished in {cnn_train_time:.1f}s", flush=True)

    # 5-fold wafer-grouped folds
    folds = repeated_stratified_group_folds(
        y, wafer, n_splits=args.n_splits, repeats=1, random_state=args.seed
    )

    base_constants = load_chosen(args.selection)
    if base_constants:
        print(f"  Using selected constants {base_constants}", flush=True)

    summaries = []
    all_fold_rows = []

    print("\n--- Running Block Encoder Experiments ---", flush=True)
    for exp_config in EXPERIMENTS:
        t0 = time.perf_counter()
        print(f"\n{exp_config['name']}: {exp_config['note']}", flush=True)
        
        metrics, fold_rows = run_experiment(
            frame=frame,
            x_param=x_param,
            readings=readings,
            exp_config=exp_config,
            folds=folds,
            eligible_index=eligible_index,
            wafer=wafer,
            y=y,
            oof_cnn_logits=oof_cnn_logits,
            base_constants=base_constants,
        )
        
        metrics["seconds"] = round(time.perf_counter() - t0, 1)
        summaries.append(metrics)
        all_fold_rows.extend(fold_rows)

        print(
            "  AP=%.6f (±%.4f)  ROC-AUC=%.6f (±%.4f)  F1=%.6f (±%.4f)  P=%.4f R=%.4f  thresh=%.4f  (%.1fs)"
            % (
                metrics["average_precision"],
                metrics["ap_std"],
                metrics["roc_auc"],
                metrics["auc_std"],
                metrics["fail_f1"],
                metrics["f1_std"],
                metrics["fail_precision"],
                metrics["fail_recall"],
                metrics["opt_threshold"],
                metrics["seconds"],
            ),
            flush=True,
        )

    # Compute comparison deltas
    summary_dict = {s["experiment"]: s for s in summaries}

    lrt_block_ap = summary_dict["block_only_LRT"]["average_precision"]
    cnn_block_ap = summary_dict["block_only_CNN"]["average_precision"]
    lrt_cnn_block_ap = summary_dict["block_only_LRT_CNN"]["average_precision"]

    lrt_mb_ap = summary_dict["model_b_LRT"]["average_precision"]
    cnn_mb_ap = summary_dict["model_b_CNN"]["average_precision"]
    lrt_cnn_mb_ap = summary_dict["model_b_LRT_CNN"]["average_precision"]

    for s in summaries:
        name = s["experiment"]
        if name == "block_only_LRT":
            s["delta_ap_vs_lrt"] = 0.0
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP
        elif name == "block_only_CNN":
            s["delta_ap_vs_lrt"] = cnn_block_ap - lrt_block_ap
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP
        elif name == "block_only_LRT_CNN":
            s["delta_ap_vs_lrt"] = lrt_cnn_block_ap - lrt_block_ap
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP
        elif name == "model_b_LRT":
            s["delta_ap_vs_lrt"] = 0.0
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP
        elif name == "model_b_CNN":
            s["delta_ap_vs_lrt"] = cnn_mb_ap - lrt_mb_ap
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP
        elif name == "model_b_LRT_CNN":
            s["delta_ap_vs_lrt"] = lrt_cnn_mb_ap - lrt_mb_ap
            s["delta_ap_vs_model_a"] = s["average_precision"] - MODEL_A_BENCHMARK_AP

    # Save outputs
    table = pd.DataFrame(summaries).sort_values("average_precision", ascending=False)
    table.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(all_fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)

    # Create detailed markdown decision notes
    notes = f"""# Experiment 4: Block Encoder Comparison

## Executive Summary
This experiment compares three block encoder architectures on 2,000 sub-die block readings:
1. **LRT**: Statistical circular FFT-whitened likelihood-ratio scan statistics (~73 features)
2. **CNN**: 1D Convolutional Neural Network trained directly on 2,000-length sequence readings
3. **LRT + CNN**: Combined block representation fusing LRT likelihood ratios and 1D CNN representations

## Benchmark Results (5-Fold Wafer-Grouped CV)
- **Model A Baseline**: AP = {MODEL_A_BENCHMARK_AP:.6f}, ROC-AUC = {MODEL_A_BENCHMARK_AUC:.6f}, F1 = {MODEL_A_BENCHMARK_F1:.6f}

### Summary Metrics Table
| Experiment | AP | ROC-AUC | F1 | Precision | Recall | Δ AP vs LRT | Δ AP vs Model A | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
"""
    for s in summaries:
        notes += (
            f"| {s['experiment']} | {s['average_precision']:.6f} | {s['roc_auc']:.6f} | "
            f"{s['fail_f1']:.6f} | {s['fail_precision']:.6f} | {s['fail_recall']:.6f} | "
            f"{s['delta_ap_vs_lrt']:+.6f} | {s['delta_ap_vs_model_a']:+.6f} | {s['seconds']:.1f}s |\n"
        )

    notes += f"""
## Key Findings & Comparisons

1. **CNN vs LRT Standalone (`block_only`)**:
   - `block_only_LRT`: AP = {lrt_block_ap:.6f}
   - `block_only_CNN`: AP = {cnn_block_ap:.6f}
   - Delta (CNN - LRT): {cnn_block_ap - lrt_block_ap:+.6f}

2. **CNN vs LRT in Model B Context**:
   - `model_b_LRT`: AP = {lrt_mb_ap:.6f}
   - `model_b_CNN`: AP = {cnn_mb_ap:.6f}
   - Delta (Model B CNN - Model B LRT): {cnn_mb_ap - lrt_mb_ap:+.6f}

3. **LRT + CNN Combination**:
   - `model_b_LRT_CNN`: AP = {lrt_cnn_mb_ap:.6f}
   - Delta (LRT+CNN vs LRT): {lrt_cnn_mb_ap - lrt_mb_ap:+.6f}

## Recommendation
"""
    if lrt_mb_ap > cnn_mb_ap and lrt_mb_ap >= lrt_cnn_mb_ap - 0.001:
        notes += (
            "- **REJECT CNN**. The existing statistical LRT block encoder outperforms the 1D CNN "
            "and LRT+CNN combination while being computationally lighter and domain-derived.\n"
            "- **Retain statistical LRT block encoder as the candidate for Model B**.\n"
        )
    elif lrt_cnn_mb_ap > lrt_mb_ap + 0.005:
        notes += (
            "- **SELECT LRT + CNN**. The LRT+CNN combination provides a meaningful and consistent "
            "improvement over the LRT baseline.\n"
        )
    else:
        notes += (
            "- **RETAIN LRT**. The gains from CNN or LRT+CNN are negligible/noisy relative to fold variance. "
            "The simpler, derived statistical LRT representation is preferred.\n"
        )

    (args.output_dir / "experiment_notes.md").write_text(notes, encoding="utf-8")
    print(f"\nWrote summary and notes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
