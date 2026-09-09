"""Data Loader module for SanDisk Yield AI Interactive Explorer.

Reads pre-computed interpretability and validation artifacts safely.
DO NOT modify production ML models or retrain anything.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_INTERPRETABILITY_DIR = Path("results/interpretability")
RESULTS_VALIDATION_DIR = Path("results/final_validation")
PLOTS_DIR = RESULTS_INTERPRETABILITY_DIR / "plots"

def safe_cache_data(func):
    """Cache data conditionally when running inside active Streamlit runtime session."""
    try:
        if hasattr(st, "runtime") and st.runtime.exists():
            return st.cache_data(func)
    except Exception:
        pass
    return func

@safe_cache_data
def load_final_metrics() -> dict:
    """Load final validation benchmark metrics."""
    metrics_path = RESULTS_VALIDATION_DIR / "final_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}

@safe_cache_data
def load_wafer_heatmap_data() -> pd.DataFrame:
    """Load pre-computed wafer heatmap grid data for showcase wafers."""
    csv_path = RESULTS_INTERPRETABILITY_DIR / "wafer_heatmap_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_all_prediction_wafers() -> pd.DataFrame:
    """Load prediction die grids for ALL 40 test/prediction wafers."""
    sub_path = Path("submission.csv")
    predict_cache_dir = Path("input/cache_predict")
    
    if sub_path.exists():
        sub_df = pd.read_csv(sub_path)
        
        # Add old_label from input/cache_predict if available
        try:
            from tuned.cache import load as load_cache
            pred_cache = load_cache(predict_cache_dir)
            if "old_label" in pred_cache.columns and len(pred_cache) == len(sub_df):
                sub_df["old_label"] = pred_cache["old_label"].values
            else:
                sub_df["old_label"] = 0
        except Exception:
            sub_df["old_label"] = 0
            
        # Merge continuous float probabilities for showcase wafers where available
        heatmap_df = load_wafer_heatmap_data()
        if not heatmap_df.empty:
            sub_df = sub_df.merge(
                heatmap_df[["wafer_id", "die_row", "die_col", "predicted_probability"]],
                on=["wafer_id", "die_row", "die_col"],
                how="left"
            )
            sub_df["predicted_probability"] = sub_df["predicted_probability"].fillna(sub_df["predicted_label"].astype(float))
        else:
            sub_df["predicted_probability"] = sub_df["predicted_label"].astype(float)
            
        return sub_df
        
    return load_wafer_heatmap_data()

@safe_cache_data
def load_explanation_samples() -> list[dict]:
    """Load structured explanation samples JSON."""
    json_path = RESULTS_INTERPRETABILITY_DIR / "explanation_samples.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return []

@safe_cache_data
def load_production_model_b():
    """Load trained production Model B champion Fusion model."""
    import joblib
    model_path = Path("results/tuned_final/model_b.joblib")
    if not model_path.exists():
        return None
    data = joblib.load(model_path)
    model = data["model"] if isinstance(data, dict) and "model" in data else data
    if hasattr(model, "block_model_") and model.block_model_ is not None:
        try:
            clf = model.block_model_.named_steps["classifier"]
            if not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
        except Exception:
            pass
    return model

@safe_cache_data
def find_wafer_parquet_path(wafer_id: str) -> Path | None:
    """Find cached parquet filepath for a specific wafer_id."""
    for base in [Path("input/cache_predict"), Path("input/cache_test"), Path("input/cache_train")]:
        if base.exists():
            matches = list(base.glob(f"*{wafer_id}.parquet"))
            if matches:
                return matches[0]
    matches = list(Path("input").rglob(f"*{wafer_id}.parquet"))
    if matches:
        return matches[0]
    return None

@safe_cache_data
def load_wafer_parquet(wafer_id: str) -> pd.DataFrame:
    """Load single wafer parquet dataframe."""
    p_path = find_wafer_parquet_path(wafer_id)
    if p_path and p_path.exists():
        return pd.read_parquet(p_path)
    return pd.DataFrame()

def get_dynamic_explanation(wafer_id: str, die_row: int, die_col: int) -> dict | None:
    """Calculate exact dynamic logit evidence decomposition for an arbitrary die from saved model artifacts."""
    model = load_production_model_b()
    if model is None:
        return None

    frame = load_wafer_parquet(wafer_id)
    if frame.empty:
        return None

    match = frame[(frame["die_row"] == die_row) & (frame["die_col"] == die_col)]
    if match.empty:
        return None

    row_idx = int(match.index[0])
    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    if not feature_cols:
        return None

    x_wafer = frame[feature_cols].to_numpy(dtype=np.float32)

    from tuned.interpretability import decompose_logit, explain_parametric, explain_spatial

    decomp = decompose_logit(model, frame, x_wafer, row_idx)

    # Tight tolerance consistency check: Final logit vs component sum
    calc_sum = decomp.c_spatial + decomp.c_param + decomp.c_block
    logit_diff = abs(calc_sum - decomp.final_logit)
    if logit_diff > 1e-4:
        import sys
        print(f"[WARNING] Logit identity mismatch for {wafer_id} ({die_row},{die_col}): sum={calc_sum}, final={decomp.final_logit}, diff={logit_diff}", file=sys.stderr)

    param_exp = explain_parametric(model.diagonal_, x_wafer[row_idx], top_k=10)
    spatial_exp = explain_spatial(model, frame, row_idx)

    # Block Context
    block_context = {
        "peak_anomaly_index": None,
        "peak_anomaly_range": None,
        "peak_scan_value": None,
        "top_block_features": [],
    }

    # Check block_anomaly_samples.csv for optional sequence peak localization metadata
    anomaly_df = load_block_anomaly_samples()
    if not anomaly_df.empty:
        anom_match = anomaly_df[
            (anomaly_df["wafer_id"] == wafer_id) &
            (anomaly_df["die_row"] == die_row) &
            (anomaly_df["die_col"] == die_col)
        ]
        if not anom_match.empty:
            anom_row = anom_match.iloc[0]
            block_context["peak_anomaly_index"] = int(anom_row.get("peak_anomaly_index", 0))
            block_context["peak_anomaly_range"] = str(anom_row.get("peak_anomaly_range", "N/A"))
            val = float(anom_row.get("peak_scan_value", anom_row.get("peak_anomaly_score", 0.0)))
            block_context["peak_scan_value"] = val

    # Compute top 5 block LRT feature attributions from 73 block columns in parquet if available
    if hasattr(model, "use_block") and model.use_block and model.block_model_ is not None:
        try:
            block_names = model.block_names_
            block_series = frame.loc[row_idx, block_names].infer_objects().replace([np.inf, -np.inf], np.nan)
            block_df = pd.DataFrame([block_series])

            clf = model.block_model_.named_steps["classifier"]
            scaler = model.block_model_.named_steps["scale"]
            imputer = model.block_model_.named_steps["imputer"]

            imputed = imputer.transform(block_df)
            scaled = scaler.transform(imputed)
            coefs = clf.coef_[0]

            feature_contribs = scaled[0] * coefs
            df_block_features = pd.DataFrame({
                "feature_name": block_names,
                "raw_value": imputed[0],
                "scaled_value": scaled[0],
                "coefficient": coefs,
                "contribution": feature_contribs,
                "abs_contribution": np.abs(feature_contribs),
            }).sort_values("abs_contribution", ascending=False)

            block_context["top_block_features"] = df_block_features.head(5).to_dict(orient="records")
        except Exception:
            pass

    old_lbl = int(frame.loc[row_idx, "old_label"]) if "old_label" in frame.columns else 0

    return {
        "die_index": row_idx,
        "wafer_id": str(wafer_id),
        "die_row": int(die_row),
        "die_col": int(die_col),
        "old_label": old_lbl,
        "predicted_probability": float(decomp.final_prob),
        "final_logit": float(decomp.final_logit),
        "logit_decomposition": {
            "spatial_contribution": float(decomp.c_spatial),
            "parametric_contribution": float(decomp.c_param),
            "block_contribution": float(decomp.c_block),
            "evidence_offset": float(decomp.offset),
            "sum_check_residual": float(decomp.residual),
        },
        "top_parametric_features": param_exp.to_dict(orient="records"),
        "spatial_context": spatial_exp,
        "block_context": block_context,
        "is_dynamic": True,
    }

def find_explanation(wafer_id: str, die_row: int, die_col: int) -> dict | None:
    """Find pre-computed explanation sample or dynamically compute exact evidence decomposition."""
    # 1. Check pre-computed explanation samples (Case 1: Representative Cases)
    explanations_json = load_explanation_samples()
    for exp in explanations_json:
        if exp["wafer_id"] == wafer_id and int(exp["die_row"]) == int(die_row) and int(exp["die_col"]) == int(die_col):
            return exp

    # 2. Dynamically compute explanation from saved production model artifacts (Case 2: Arbitrary Dies)
    return get_dynamic_explanation(wafer_id, die_row, die_col)

@safe_cache_data
def load_block_anomaly_samples() -> pd.DataFrame:
    """Load block anomaly summary samples."""
    csv_path = RESULTS_INTERPRETABILITY_DIR / "block_anomaly_samples.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_contribution_summary() -> pd.DataFrame:
    """Load logit contribution summary CSV."""
    csv_path = RESULTS_INTERPRETABILITY_DIR / "contribution_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_model_a_predictions() -> pd.DataFrame:
    """Load pre-computed Model A die-level probability predictions for test wafers."""
    csv_path = Path("results/model_a_wafer_predictions.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_model_a_b_joined_predictions() -> pd.DataFrame:
    """Load and join Model A and Model B die-level predictions for all test wafers.

    Returns DataFrame with columns:
    - wafer_id, die_row, die_col
    - prob_model_a
    - prob_model_b
    - risk_difference = prob_model_b - prob_model_a
    """
    df_a = load_model_a_predictions()
    df_b = load_all_prediction_wafers()
    if df_a.empty or df_b.empty:
        return pd.DataFrame()
    
    df_a_renamed = df_a.rename(columns={"predicted_probability": "prob_model_a"})
    df_b_renamed = df_b.rename(columns={"predicted_probability": "prob_model_b"})
    
    merged = df_a_renamed.merge(
        df_b_renamed[["wafer_id", "die_row", "die_col", "prob_model_b"]],
        on=["wafer_id", "die_row", "die_col"],
        how="inner"
    )
    merged["risk_difference"] = merged["prob_model_b"] - merged["prob_model_a"]
    return merged

def get_plot_path(filename: str) -> Path | None:
    """Get absolute path to pre-generated plot PNG."""
    plot_file = PLOTS_DIR / filename
    if plot_file.exists():
        return plot_file
    return None

def format_percent(val: float) -> str:
    """Format float probability to clean human-readable percentage."""
    if val >= 0.9999:
        return "100.0%"
    elif val <= 0.0001:
        return "0.01%"
    else:
        return f"{val * 100:.1f}%"

def format_logit(val: float) -> str:
    """Format float logit to signed 2-decimal string."""
    return f"{val:+.2f}"

@safe_cache_data
def load_signature_summary() -> dict:
    """Load JSON summary for failure signature discovery."""
    json_path = Path("results/failure_signatures/signature_summary.json")
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {}

@safe_cache_data
def load_cluster_summary() -> pd.DataFrame:
    """Load cluster summary CSV for failure signatures."""
    csv_path = Path("results/failure_signatures/cluster_summary.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_cluster_profiles() -> pd.DataFrame:
    """Load normalized cluster evidence profiles CSV."""
    csv_path = Path("results/failure_signatures/cluster_profiles.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def load_cluster_assignments() -> pd.DataFrame:
    """Load die-level cluster assignments CSV."""
    csv_path = Path("results/failure_signatures/cluster_assignments.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@safe_cache_data
def get_cluster_wafers_summary(cluster_id: int) -> pd.DataFrame:
    """Aggregate dies belonging to selected cluster by wafer.

    Returns DataFrame with columns:
    - wafer_id
    - signature_die_count
    - pct_of_wafer_dies
    - mean_model_b_prob
    - mean_model_a_prob
    - mean_risk_diff
    - descriptive_failure_rate
    - n_failures
    """
    df = load_cluster_assignments()
    if df.empty or "cluster_id" not in df.columns:
        return pd.DataFrame()

    # Pre-calculate total dies per wafer across dataset
    wafer_totals = df.groupby("wafer_id")["die_id"].count().to_dict()

    c_df = df[df["cluster_id"] == cluster_id].copy()
    if c_df.empty:
        return pd.DataFrame()

    rows = []
    for w_id, w_sub in c_df.groupby("wafer_id"):
        sig_count = len(w_sub)
        tot_count = wafer_totals.get(w_id, sig_count)
        pct_wafer = (sig_count / tot_count) * 100.0 if tot_count > 0 else 0.0
        n_fail = int(w_sub["label"].sum()) if "label" in w_sub.columns else 0
        fail_rate = (n_fail / sig_count) * 100.0 if sig_count > 0 else 0.0

        mean_pb = float(w_sub["model_b_probability"].mean()) if "model_b_probability" in w_sub.columns else 0.0
        mean_pa = float(w_sub["model_a_probability"].mean()) if "model_a_probability" in w_sub.columns else 0.0
        mean_diff = float(w_sub["risk_difference"].mean()) if "risk_difference" in w_sub.columns else (mean_pb - mean_pa)

        rows.append({
            "wafer_id": w_id,
            "signature_die_count": sig_count,
            "total_wafer_dies": tot_count,
            "pct_of_wafer_dies": pct_wafer,
            "mean_model_b_prob": mean_pb,
            "mean_model_a_prob": mean_pa,
            "mean_risk_diff": mean_diff,
            "descriptive_failure_rate": fail_rate,
            "n_failures": n_fail,
        })

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(
            by=["signature_die_count", "mean_model_b_prob"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return res_df

@safe_cache_data
def get_cluster_wafer_dies(cluster_id: int, wafer_id: str) -> pd.DataFrame:
    """Filter dies from selected wafer belonging to selected cluster.

    Returns DataFrame with columns:
    - wafer_id, die_row, die_col, die_id
    - model_a_probability, model_b_probability, risk_difference
    - label, old_label, parametric_evidence, spatial_evidence, block_evidence
    - signature_name, dominant_source
    """
    df = load_cluster_assignments()
    if df.empty or "cluster_id" not in df.columns or "wafer_id" not in df.columns:
        return pd.DataFrame()

    c_df = df[(df["cluster_id"] == cluster_id) & (df["wafer_id"] == wafer_id)].copy()
    if c_df.empty:
        return pd.DataFrame()

    summary_df = load_cluster_summary()
    if not summary_df.empty and "cluster_id" in summary_df.columns:
        sig_info = summary_df[summary_df["cluster_id"] == cluster_id]
        if not sig_info.empty:
            c_df["signature_name"] = sig_info.iloc[0].get("signature_name", "Cluster " + str(cluster_id))
            c_df["dominant_source"] = sig_info.iloc[0].get("dominant_source", "Unknown")

    c_df = c_df.sort_values(by="model_b_probability", ascending=False).reset_index(drop=True)
    return c_df

@safe_cache_data
def detect_wafer_hotspots(
    wafer_id: str,
    percentile_threshold: float = 95.0,
    min_hotspot_dies: int = 3,
    risk_col: str = "prob_model_b",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect spatial high-risk hotspots on a single wafer using 8-neighbor connectivity.

    Operates strictly on pre-computed prediction risk surfaces without ground-truth labels.

    Returns:
    - hotspots_summary_df: DataFrame of ranked hotspots with summary statistics
    - annotated_wafer_df: Full wafer die grid DataFrame with assigned `hotspot_id`
    """
    df = load_model_a_b_joined_predictions()
    if df.empty or "wafer_id" not in df.columns or wafer_id not in df["wafer_id"].values:
        df_all = load_all_prediction_wafers()
        if not df_all.empty and "wafer_id" in df_all.columns and wafer_id in df_all["wafer_id"].values:
            df = df_all.copy()
            if "predicted_probability" in df.columns and "prob_model_b" not in df.columns:
                df["prob_model_b"] = df["predicted_probability"]
            if "prob_model_a" not in df.columns:
                df["prob_model_a"] = 0.0
            if "risk_difference" not in df.columns:
                df["risk_difference"] = df["prob_model_b"] - df["prob_model_a"]
        else:
            df_cluster = load_cluster_assignments()
            if not df_cluster.empty and "wafer_id" in df_cluster.columns and wafer_id in df_cluster["wafer_id"].values:
                df = df_cluster.copy()
                if "model_b_probability" in df.columns and "prob_model_b" not in df.columns:
                    df["prob_model_b"] = df["model_b_probability"]
                if "model_a_probability" in df.columns and "prob_model_a" not in df.columns:
                    df["prob_model_a"] = df["model_a_probability"]
                if "risk_difference" not in df.columns:
                    df["risk_difference"] = df["prob_model_b"] - df["prob_model_a"]

    if df.empty or "wafer_id" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    wafer_df = df[df["wafer_id"] == wafer_id].copy()
    if wafer_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if risk_col not in wafer_df.columns:
        risk_col = "prob_model_b" if "prob_model_b" in wafer_df.columns else "predicted_probability"

    risk_vals = wafer_df[risk_col].values
    if len(risk_vals) == 0:
        return pd.DataFrame(), wafer_df

    cutoff = float(np.percentile(risk_vals, percentile_threshold))
    high_risk_mask = wafer_df[risk_col] >= cutoff
    high_risk_df = wafer_df[high_risk_mask]
    
    if high_risk_df.empty:
        wafer_df["hotspot_id"] = None
        wafer_df["is_high_risk_die"] = False
        wafer_df["is_hotspot_die"] = False
        wafer_df["is_isolated_high_risk"] = False
        return pd.DataFrame(), wafer_df

    coord_to_idx = {
        (int(row["die_row"]), int(row["die_col"])): idx 
        for idx, row in high_risk_df.iterrows()
    }
    
    visited = set()
    raw_clusters = []
    
    neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for coord in coord_to_idx:
        if coord in visited:
            continue
        
        cluster_coords = []
        queue = [coord]
        visited.add(coord)
        
        while queue:
            curr_r, curr_c = queue.pop(0)
            cluster_coords.append((curr_r, curr_c))
            
            for dr, dc in neighbors_offsets:
                nr, nc = curr_r + dr, curr_c + dc
                neighbor_coord = (nr, nc)
                if neighbor_coord in coord_to_idx and neighbor_coord not in visited:
                    visited.add(neighbor_coord)
                    queue.append(neighbor_coord)
                    
        raw_clusters.append(cluster_coords)

    cluster_stats_temp = []
    for coords in raw_clusters:
        indices = [coord_to_idx[c] for c in coords]
        sub = wafer_df.loc[indices]
        m_risk = float(sub[risk_col].mean())
        cluster_stats_temp.append((m_risk, len(coords), coords, indices))
        
    cluster_stats_temp.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    wafer_df["hotspot_id"] = None
    wafer_df["is_high_risk_die"] = high_risk_mask
    wafer_df["is_hotspot_die"] = False
    wafer_df["is_isolated_high_risk"] = False

    official_hotspots_data = []
    hotspot_counter = 1

    for m_risk, n_dies, coords, indices in cluster_stats_temp:
        if n_dies >= min_hotspot_dies:
            h_id = f"H{hotspot_counter}"
            hotspot_counter += 1
            
            wafer_df.loc[indices, "hotspot_id"] = h_id
            wafer_df.loc[indices, "is_hotspot_die"] = True
            
            sub = wafer_df.loc[indices]
            r_mins, r_maxs = int(sub["die_row"].min()), int(sub["die_row"].max())
            c_mins, c_maxs = int(sub["die_col"].min()), int(sub["die_col"].max())
            cent_r = float(sub["die_row"].mean())
            cent_c = float(sub["die_col"].mean())
            
            p_b_mean = float(sub["prob_model_b"].mean()) if "prob_model_b" in sub.columns else m_risk
            p_a_mean = float(sub["prob_model_a"].mean()) if "prob_model_a" in sub.columns else 0.0
            p_diff_mean = float(sub["risk_difference"].mean()) if "risk_difference" in sub.columns else (p_b_mean - p_a_mean)

            official_hotspots_data.append({
                "hotspot_id": h_id,
                "wafer_id": wafer_id,
                "n_dies": n_dies,
                "mean_risk": m_risk,
                "max_risk": float(sub[risk_col].max()),
                "median_risk": float(sub[risk_col].median()),
                "min_risk": float(sub[risk_col].min()),
                "mean_model_a_prob": p_a_mean,
                "mean_model_b_prob": p_b_mean,
                "mean_risk_diff": p_diff_mean,
                "row_min": r_mins,
                "row_max": r_maxs,
                "col_min": c_mins,
                "col_max": c_maxs,
                "centroid_row": cent_r,
                "centroid_col": cent_c,
                "center_str": f"({cent_r:.1f}, {cent_c:.1f})",
                "bounding_region": f"Row [{r_mins}, {r_maxs}], Col [{c_mins}, {c_maxs}]",
            })
        else:
            wafer_df.loc[indices, "hotspot_id"] = "Isolated"
            wafer_df.loc[indices, "is_isolated_high_risk"] = True

    hotspots_summary_df = pd.DataFrame(official_hotspots_data)
    return hotspots_summary_df, wafer_df
