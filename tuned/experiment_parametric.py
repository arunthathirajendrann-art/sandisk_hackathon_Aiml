"""Experiment Runner for Parametric Representation Evaluation.

Compares:
1. parametric_only_baseline (DiagonalScore Naive Bayes)
2. parametric_only_mlp (PyTorch 500->256->128->64->1 MLP Encoder)
3. model_a_baseline (Model A with Naive Bayes DiagonalScore)
4. model_a_mlp (Model A with PyTorch MLP Encoder)

Uses exact wafer-grouped StratifiedGroupKFold cross-validation (5 splits x 3 repeats = 15 folds).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    repeated_stratified_group_folds,
)
from tuned.cache import eligible_mask, load
from tuned.pipeline import Fusion


@dataclass(frozen=True)
class ParametricExperiment:
    name: str
    note: str
    options: dict = field(default_factory=dict)


EXPERIMENTS = (
    ParametricExperiment(
        "parametric_only_baseline",
        "Die measurements alone via Naive Bayes DiagonalScore",
        dict(use_block=False, correct_prior=False, use_rate=False, use_hazard=False, subtract_density=False),
    ),
    ParametricExperiment(
        "parametric_only_mlp",
        "Die measurements alone via PyTorch MLP Encoder (500->256->128->64->1)",
        dict(use_block=False, correct_prior=False, use_rate=False, use_hazard=False, use_parametric_mlp=True, subtract_density=False),
    ),
    ParametricExperiment(
        "model_a_baseline",
        "Model A (Spatial Context + DiagonalScore + Rate Posterior)",
        dict(use_block=False, subtract_density=False),
    ),
    ParametricExperiment(
        "model_a_mlp",
        "Model A (Spatial Context + PyTorch MLP Encoder + Rate Posterior)",
        dict(use_block=False, use_parametric_mlp=True, subtract_density=False),
    ),
)


def run_experiment(frame: pd.DataFrame, x: np.ndarray, exp: ParametricExperiment, folds, eligible_index, wafer, y):
    started = time.perf_counter()
    oof_predictions = np.zeros(len(eligible_index), dtype=np.float64)
    seen_counts = np.zeros(len(eligible_index), dtype=np.int16)
    fold_ap_scores = []
    fold_f1_scores = []

    for fold in folds:
        train_rows = np.isin(
            frame["wafer_id"].astype(str).to_numpy(),
            list(set(wafer[fold.train_index]))
        )
        score_rows = np.zeros(len(frame), dtype=bool)
        score_rows[eligible_index[fold.validation_index]] = True

        model = Fusion(**exp.options).fit(frame, x, train_rows)
        pred = model.predict_proba(frame, x, score_rows)

        oof_predictions[fold.validation_index] += pred
        seen_counts[fold.validation_index] += 1

        val_y = y[fold.validation_index]
        fold_thresh = best_f1_threshold(val_y, pred)
        fold_metrics = classification_metrics(val_y, pred, fold_thresh)
        fold_ap_scores.append(fold_metrics["average_precision"])
        fold_f1_scores.append(fold_metrics["fail_f1"])

    oof_predictions /= np.maximum(seen_counts, 1)
    threshold = best_f1_threshold(y, oof_predictions)
    overall_metrics = classification_metrics(y, oof_predictions, threshold)
    elapsed = time.perf_counter() - started

    summary = {
        "experiment": exp.name,
        "note": exp.note,
        "average_precision": overall_metrics["average_precision"],
        "roc_auc": overall_metrics["roc_auc"],
        "fail_f1": overall_metrics["fail_f1"],
        "fail_precision": overall_metrics["fail_precision"],
        "fail_recall": overall_metrics["fail_recall"],
        "accuracy": overall_metrics["accuracy"],
        "threshold": threshold,
        "ap_std": float(np.std(fold_ap_scores)),
        "f1_std": float(np.std(fold_f1_scores)),
        "seconds": round(elapsed, 2),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path, help="Path to input/cache_train")
    parser.add_argument("output_dir", type=Path, help="Path to results/parametric_encoder")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load(args.cache_dir)

    parametric_cols = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric_cols].to_numpy(dtype=np.float32)
    frame_clean = frame.drop(columns=parametric_cols)

    eligible = eligible_mask(frame_clean)
    eligible_index = np.flatnonzero(eligible)
    y = frame_clean.loc[eligible, "label"].to_numpy(dtype=np.int8)
    wafer = frame_clean.loc[eligible, "wafer_id"].astype(str).to_numpy()

    folds = repeated_stratified_group_folds(
        y, wafer, n_splits=args.n_splits, repeats=args.repeats, random_state=args.seed
    )

    results = []
    print(f"Running Parametric Encoder Evaluation ({len(folds)} folds)...", flush=True)

    for exp in EXPERIMENTS:
        print(f"Evaluating {exp.name}...", flush=True)
        res = run_experiment(frame_clean, x, exp, folds, eligible_index, wafer, y)
        results.append(res)
        print(
            f"  {exp.name:<25} AP={res['average_precision']:.4f} "
            f"ROC-AUC={res['roc_auc']:.4f} F1={res['fail_f1']:.4f} "
            f"P={res['fail_precision']:.4f} R={res['fail_recall']:.4f} ({res['seconds']:.1f}s)"
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nFinal Parametric Experiment Summary:")
    print(res_df[["experiment", "average_precision", "roc_auc", "fail_f1", "fail_precision", "fail_recall", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
