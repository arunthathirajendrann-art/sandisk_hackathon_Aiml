"""Experiment 3: Graph Neural Network (GNN) Spatial Representation Screening (5-fold Wafer-Grouped CV).

Compares Model A Baseline vs Model A + WaferGNN Learned Spatial Representation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from tuned import cache
from tuned.pipeline import Fusion


def evaluate_5fold(
    frame: pd.DataFrame,
    x: np.ndarray,
    model_kwargs: dict,
    name: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    wafer_labels = frame.groupby("wafer_id")["label"].max().to_numpy()
    wafer_ids = frame.groupby("wafer_id")["label"].max().index.to_numpy()
    
    oof_probs = np.zeros(len(frame), dtype=np.float64)
    eligible = frame["old_label"].to_numpy() == 0
    fold_aps = []
    fold_f1s = []

    start_time = time.perf_counter()
    
    for fold, (train_w_idx, val_w_idx) in enumerate(
        sgkf.split(wafer_ids, wafer_labels, groups=wafer_ids)
    ):
        train_wafers = set(wafer_ids[train_w_idx])
        val_wafers = set(wafer_ids[val_w_idx])

        train_mask = frame["wafer_id"].isin(train_wafers).to_numpy()
        val_mask = frame["wafer_id"].isin(val_wafers).to_numpy()

        val_eligible = val_mask & eligible

        model = Fusion(**model_kwargs, random_state=seed + fold)
        model.fit(frame, x, train_mask)

        probs = model.predict_proba(frame, x, val_eligible)
        oof_probs[val_eligible] = probs

        y_val_elig = frame.loc[val_eligible, "label"].to_numpy()
        if len(np.unique(y_val_elig)) > 1:
            fold_ap = float(average_precision_score(y_val_elig, probs))
            fold_aps.append(fold_ap)

    elapsed = time.perf_counter() - start_time
    
    y_true = frame.loc[eligible, "label"].to_numpy()
    y_probs = oof_probs[eligible]

    ap = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs))

    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1, best_thresh, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresh = float(th)
            best_prec = float(p)
            best_rec = float(r)

    for fold, (train_w_idx, val_w_idx) in enumerate(
        sgkf.split(wafer_ids, wafer_labels, groups=wafer_ids)
    ):
        val_wafers = set(wafer_ids[val_w_idx])
        val_eligible = frame["wafer_id"].isin(val_wafers).to_numpy() & eligible
        y_val = frame.loc[val_eligible, "label"].to_numpy()
        p_val = oof_probs[val_eligible]
        preds = (p_val >= best_thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )
        fold_f1s.append(float(f1))

    return {
        "experiment": name,
        "average_precision": ap,
        "roc_auc": roc_auc,
        "fail_f1": best_f1,
        "fail_precision": best_prec,
        "fail_recall": best_rec,
        "accuracy": float(np.mean((y_probs >= best_thresh).astype(int) == y_true)),
        "threshold": best_thresh,
        "ap_std": float(np.std(fold_aps)) if fold_aps else 0.0,
        "f1_std": float(np.std(fold_f1s)) if fold_f1s else 0.0,
        "seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: GNN Spatial Screening")
    parser.add_argument("cache_dir", type=Path, help="Directory containing parquet wafer cache")
    parser.add_argument("output_dir", type=Path, help="Directory to save experiment results")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from cache...")
    frame = cache.load(args.cache_dir)
    feature_columns = [c for c in frame.columns if c.startswith("feature_")]
    x = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)

    experiments = [
        (
            "model_a_baseline",
            {"use_block": False, "use_parametric": True, "use_parametric_mlp": False, "use_gnn_spatial": False},
        ),
        (
            "model_a_gnn",
            {"use_block": False, "use_parametric": True, "use_parametric_mlp": False, "use_gnn_spatial": True},
        ),
        (
            "spatial_only_baseline",
            {"use_block": False, "use_parametric": False, "use_parametric_mlp": False, "use_gnn_spatial": False},
        ),
        (
            "spatial_only_gnn",
            {"use_block": False, "use_parametric": False, "use_parametric_mlp": False, "use_gnn_spatial": True},
        ),
    ]

    results = []
    print("\nRunning GNN Spatial Evaluation (5 folds)...")
    for name, kwargs in experiments:
        print(f"Evaluating {name}...")
        res = evaluate_5fold(frame, x, kwargs, name=name, n_splits=5)
        results.append(res)
        print(
            f"  {name:<27} AP={res['average_precision']:.4f} ROC-AUC={res['roc_auc']:.4f} "
            f"F1={res['fail_f1']:.4f} P={res['fail_precision']:.4f} R={res['fail_recall']:.4f} "
            f"({res['seconds']}s)"
        )

    summary_json = args.output_dir / "experiment_summary.json"
    summary_csv = args.output_dir / "experiment_summary.csv"

    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    df_res = pd.DataFrame(results)
    df_res.to_csv(summary_csv, index=False)

    print("\nFinal GNN Experiment Summary:")
    print(df_res[["experiment", "average_precision", "roc_auc", "fail_f1", "fail_precision", "fail_recall", "seconds"]].to_string())


if __name__ == "__main__":
    main()
