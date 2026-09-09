"""Multi-Resolution Interpretability module for SanDisk Yield AI Model B.

Decomposes Model B predictions into exact additive component log-odds contributions,
feature-level parametric attributions, spatial context factors, and 2,000-reading block sequence anomaly localizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tuned import blocks, pipeline
from tuned.pipeline import Fusion, CAP, _logit


@dataclass
class LogitDecomposition:
    """Exact additive decomposition of Model B's log-odds prediction."""

    final_prob: float
    final_logit: float
    c_spatial: float
    c_param: float
    c_block: float
    offset: float
    residual: float


def decompose_logit(model: Fusion, frame: pd.DataFrame, x: np.ndarray, row_idx: int) -> LogitDecomposition:
    """Decomposes Model B's predicted log-odds into exact additive spatial, parametric, and block contributions."""
    row_mask = np.zeros(len(frame), dtype=bool)
    row_mask[row_idx] = True

    design = model._design(frame, x, row_mask)
    parts = model.head_.partial(design)
    zero = np.zeros(len(design))

    e_param = float(parts.get("parametric_score", zero)[0])
    e_block = float(parts.get("block_score", zero)[0])

    correction, _ = model._split(design)
    hazard = float(model._shape(frame, row_mask)[0] * np.exp(correction[0]))

    wafer = str(frame.loc[row_idx, "wafer_id"])
    wafer_rate = float(model.fitted_rates_.get(wafer, model.overall_rate_))

    prior_p = min(max(wafer_rate * hazard, 1e-9), CAP)
    c_spatial = float(_logit(prior_p))

    c_param = e_param - 0.5 * model.evidence_offset_
    c_block = e_block - 0.5 * model.evidence_offset_
    offset = float(model.evidence_offset_)

    calc_logit = c_spatial + e_param + e_block - offset
    final_prob = float(1.0 / (1.0 + np.exp(-calc_logit)))

    model_prob = float(model.predict_proba(frame, x, row_mask)[0])
    residual = abs(final_prob - model_prob)

    return LogitDecomposition(
        final_prob=final_prob,
        final_logit=calc_logit,
        c_spatial=c_spatial,
        c_param=c_param,
        c_block=c_block,
        offset=offset,
        residual=residual,
    )


def explain_parametric(diagonal_score, x_row: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    """Computes exact per-feature contributions from DiagonalScore for the 500 parametric measurements."""
    weights = diagonal_score.weights.astype(np.float64)
    centre = diagonal_score.centre.astype(np.float64)
    scale = float(diagonal_score.scale)

    x_val = np.asarray(x_row, dtype=np.float64)
    contributions = (x_val - centre) * weights / scale

    df = pd.DataFrame({
        "feature_index": np.arange(len(weights)),
        "feature_name": [f"feature_{i}" for i in range(len(weights))],
        "feature_value": x_val,
        "weight": weights,
        "centre": centre,
        "contribution": contributions,
        "abs_contribution": np.abs(contributions),
        "direction": np.where(contributions >= 0, "+", "-"),
    }).sort_values("abs_contribution", ascending=False)

    return df.head(top_k).reset_index(drop=True)


def explain_spatial(model: Fusion, frame: pd.DataFrame, row_idx: int) -> dict:
    """Exposes spatial prior covariates and their partial contributions."""
    row_mask = np.zeros(len(frame), dtype=bool)
    row_mask[row_idx] = True

    design = model._design(frame, x=np.zeros((len(frame), 500), dtype=np.float32), rows=row_mask)
    parts = model.head_.partial(design)

    spatial_covariates = {}
    for name in model.prior_linear_:
        if name in frame.columns:
            val = float(frame.loc[row_idx, name])
            part_val = float(parts.get(name, [0.0])[0])
            spatial_covariates[name] = {"value": val, "partial_contribution": part_val}

    hazard_shape = float(frame.loc[row_idx, pipeline.HAZARD_SHAPE]) if pipeline.HAZARD_SHAPE in frame.columns else 1.0
    wafer = str(frame.loc[row_idx, "wafer_id"])
    wafer_rate = float(model.fitted_rates_.get(wafer, model.overall_rate_))

    die_r = int(frame.loc[row_idx, "die_row"]) if "die_row" in frame.columns else None
    die_c = int(frame.loc[row_idx, "die_col"]) if "die_col" in frame.columns else None

    return {
        "wafer_id": wafer,
        "die_row": die_r,
        "die_col": die_c,
        "hazard_shape": hazard_shape,
        "wafer_rate": wafer_rate,
        "spatial_covariates": spatial_covariates,
    }


def explain_block(model: Fusion, frame: pd.DataFrame, readings_row: np.ndarray, row_idx: int) -> dict:
    """Exposes top block LRT feature contributions and localizes sequence anomaly peak index."""
    if not model.use_block or model.block_model_ is None:
        return {"note": "Block LRT encoder not enabled."}

    # Extract single die block features
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

    # Localize sequence anomaly peak from 2,000 block sequence readings
    u_seq = np.asarray(readings_row, dtype=np.float64)
    noise_model = blocks.NoiseModel.fit(u_seq.reshape(1, -1))
    
    # Compute circular scan profile
    k = noise_model.k
    centred = (u_seq.reshape(1, -1) - noise_model.level) / noise_model.scale
    spectrum = np.fft.rfft(centred, axis=1)
    
    # Use standard envelope width 90
    envelope_90 = np.conj(np.fft.rfft(blocks._envelope(k, 90)))
    gamma = 0.0
    u_white = np.fft.irfft(spectrum / noise_model.response[None, :] ** gamma, n=k, axis=1)
    sigma2 = float(np.median(u_white ** 2)) / blocks.CHI2_MEDIAN
    sigma = np.sqrt(sigma2)
    amplitude = 0.45 * sigma
    tau2 = (blocks.SPIKE_SPREAD * amplitude) ** 2
    total = sigma2 + tau2
    density_ratio = np.sqrt(sigma2 / total) * np.exp(-0.5 * ((u_white - amplitude) ** 2 / total - u_white ** 2 / sigma2))
    evidence = np.fft.rfft(density_ratio - 1.0, axis=1)
    scan_profile = np.fft.irfft(evidence * envelope_90[None, :], n=k, axis=1)[0]

    peak_pos = int(np.argmax(scan_profile))
    range_start = max(0, peak_pos - 45)
    range_end = min(k - 1, peak_pos + 45)

    return {
        "top_block_features": df_block_features.head(5).to_dict(orient="records"),
        "peak_anomaly_index": peak_pos,
        "peak_anomaly_range": f"[{range_start}, {range_end}]",
        "peak_scan_value": float(scan_profile[peak_pos]),
        "scan_profile": scan_profile,
        "readings_sequence": u_seq,
    }


def explain_die(model: Fusion, frame: pd.DataFrame, x: np.ndarray, readings: np.ndarray, row_idx: int) -> dict:
    """Generates complete multi-resolution explanation for a single die."""
    decomp = decompose_logit(model, frame, x, row_idx)
    param_exp = explain_parametric(model.diagonal_, x[row_idx], top_k=10)
    spatial_exp = explain_spatial(model, frame, row_idx)
    block_exp = explain_block(model, frame, readings[row_idx], row_idx)

    die_r = int(frame.loc[row_idx, "die_row"]) if "die_row" in frame.columns else None
    die_c = int(frame.loc[row_idx, "die_col"]) if "die_col" in frame.columns else None

    return {
        "die_index": int(row_idx),
        "wafer_id": str(frame.loc[row_idx, "wafer_id"]),
        "die_row": die_r,
        "die_col": die_c,
        "old_label": int(frame.loc[row_idx, "old_label"]),
        "predicted_probability": decomp.final_prob,
        "final_logit": decomp.final_logit,
        "logit_decomposition": {
            "spatial_contribution": decomp.c_spatial,
            "parametric_contribution": decomp.c_param,
            "block_contribution": decomp.c_block,
            "evidence_offset": decomp.offset,
            "sum_check_residual": decomp.residual,
        },
        "top_parametric_features": param_exp.to_dict(orient="records"),
        "spatial_context": spatial_exp,
        "block_context": {
            "peak_anomaly_index": block_exp.get("peak_anomaly_index"),
            "peak_anomaly_range": block_exp.get("peak_anomaly_range"),
            "peak_scan_value": block_exp.get("peak_scan_value"),
            "top_block_features": block_exp.get("top_block_features"),
        },
    }


def plot_wafer_heatmap(df_wafer: pd.DataFrame, wafer_id: str, output_path: Path):
    """Renders 2D spatial risk heatmap for a wafer."""
    plt.figure(figsize=(10, 8))
    r_col = "die_row" if "die_row" in df_wafer.columns else "row"
    c_col = "die_col" if "die_col" in df_wafer.columns else "col"

    pivot_prob = df_wafer.pivot(index=r_col, columns=c_col, values="pred_prob")
    plt.imshow(pivot_prob, cmap="YlOrRd", origin="lower")
    plt.colorbar(label="Predicted Failure Probability")
    
    # Overlay old failure markers
    old_fails = df_wafer[df_wafer["old_label"] == 1]
    if len(old_fails) > 0:
        plt.scatter(old_fails[c_col], old_fails[r_col], c="blue", marker="x", s=40, label="Pre-test Old Fail")
        plt.legend()

    plt.title(f"Wafer Spatial Risk Heatmap — {wafer_id}")
    plt.xlabel("Die Column")
    plt.ylabel("Die Row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_block_sequence(readings_seq: np.ndarray, scan_profile: np.ndarray, peak_pos: int, title: str, output_path: Path):
    """Renders 2,000 block sequence readings vs circular LRT scan anomaly profile."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(readings_seq, color="gray", alpha=0.5, linewidth=0.8, label="2000 Block Readings")
    ax1.set_xlabel("Sub-die Block Sequence Position (0 to 1999)")
    ax1.set_ylabel("Raw Reading Value", color="gray")

    ax2 = ax1.twinx()
    ax2.plot(scan_profile, color="red", linewidth=1.5, label="LRT Scan Anomaly Profile")
    ax2.set_ylabel("Circular LRT Log-Likelihood Ratio", color="red")

    ax2.axvline(peak_pos, color="darkred", linestyle="--", linewidth=1.5, label=f"Peak Anomaly (Index {peak_pos})")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
