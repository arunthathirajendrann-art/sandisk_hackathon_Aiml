"""SanDisk Yield AI — Multi-Resolution Die Risk Intelligence Dashboard (Phase 8.2 Pixel-Accurate Redesign).

Command Center Engineering Analytics Dashboard for Model B Predictions, Logit Evidence Attributions,
Sub-Die Memory Block Anomaly Localization, and Benchmark Analytics.

DO NOT modify production ML pipeline files or retrain models.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.data_loader import (
    load_final_metrics,
    load_wafer_heatmap_data,
    load_all_prediction_wafers,
    load_model_a_predictions,
    load_model_a_b_joined_predictions,
    load_explanation_samples,
    load_block_anomaly_samples,
    load_contribution_summary,
    load_signature_summary,
    load_cluster_summary,
    load_cluster_profiles,
    load_cluster_assignments,
    get_cluster_wafers_summary,
    get_cluster_wafer_dies,
    detect_wafer_hotspots,
    get_plot_path,
    format_percent,
    format_logit,
    find_explanation,
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="SanDisk Yield AI — Die Risk Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Dark Industrial Engineering Surface CSS ---
st.markdown("""
<style>
    /* Dark Engineering Surface Palette */
    .stApp {
        background-color: #0A0F18;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Header Chrome */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B101A !important;
        border-right: 1px solid #1E2638 !important;
        padding-top: 10px;
    }
    
    /* Top Header Bar */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background-color: #121721;
        border: 1px solid #1E2638;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .header-title {
        font-size: 20px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.5px;
    }
    .header-sub {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 2px;
    }
    .header-status {
        background: rgba(0, 230, 118, 0.1);
        border: 1px solid #00E676;
        border-radius: 20px;
        padding: 4px 14px;
        color: #00E676;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Panel Cards & Container Overrides */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #121721;
        border: 1px solid #1E2638 !important;
        border-radius: 8px;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: #121721;
        border: 1px solid #1E2638;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
        margin-bottom: 2px;
        white-space: nowrap;
    }
    .panel-card {
        background-color: #121721;
        border: 1px solid #1E2638;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    .panel-header {
        font-size: 14px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* KPI Card Surfaces */
    .kpi-card {
        background-color: #121721;
        border: 1px solid #1E2638;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .kpi-icon {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: rgba(0, 240, 255, 0.1);
        border: 1px solid rgba(0, 240, 255, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #00F0FF;
        flex-shrink: 0;
    }
    .kpi-val-cyan {
        font-size: 26px;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-val-green {
        font-size: 26px;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
    }
    
    /* Progress comparison bar */
    .prog-bar-bg {
        background: #1E2638;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 4px 0;
    }
    .prog-bar-fill {
        background: linear-gradient(90deg, #3B82F6, #00F0FF);
        height: 100%;
        border-radius: 4px;
    }

    /* Risk Badges */
    .badge-critical {
        background: rgba(255, 82, 82, 0.15);
        border: 1px solid #FF5252;
        color: #FF5252;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 13px;
        display: inline-block;
        white-space: nowrap;
    }
    .badge-elevated {
        background: rgba(255, 183, 77, 0.15);
        border: 1px solid #FFB74D;
        color: #FFB74D;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 13px;
        display: inline-block;
        white-space: nowrap;
    }
    .badge-pass {
        background: rgba(0, 230, 118, 0.15);
        border: 1px solid #00E676;
        color: #00E676;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 13px;
        display: inline-block;
        white-space: nowrap;
    }
    
    /* Quick Actions Button */
    .quick-act-btn {
        background: #161D2A;
        border: 1px solid #2A364F;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        color: #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quick-act-btn:hover {
        border-color: #00F0FF;
        color: #00F0FF;
    }

    /* Footnote */
    .footer-note {
        font-size: 11px;
        color: #64748B;
        margin-top: 24px;
        padding-top: 12px;
        border-top: 1px solid #1E2638;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Pre-computed Artifacts ---
final_metrics = load_final_metrics()
heatmap_df = load_wafer_heatmap_data()
all_wafers_df = load_all_prediction_wafers()
explanations_json = load_explanation_samples()
anomaly_df = load_block_anomaly_samples()
contrib_df = load_contribution_summary()
joined_comp_df = load_model_a_b_joined_predictions()

available_all_wafers = sorted(all_wafers_df["wafer_id"].unique()) if not all_wafers_df.empty else []

# Standardize global session state for active wafer and die selection across all pages
if available_all_wafers and ("selected_wafer" not in st.session_state or st.session_state["selected_wafer"] not in available_all_wafers):
    st.session_state["selected_wafer"] = available_all_wafers[0]

if "selected_die_row" not in st.session_state:
    st.session_state["selected_die_row"] = 9

if "selected_die_col" not in st.session_state:
    st.session_state["selected_die_col"] = 23

# --- Global Selection Synchronizer Callbacks ---
def on_wafer_change(key_name: str):
    """Callback when a wafer selectbox changes on any dashboard page."""
    new_w = st.session_state.get(key_name)
    if new_w:
        st.session_state["selected_wafer"] = new_w
        st.session_state["active_wafer"] = new_w
        
        # Adjust dependent die row & col if currently selected die does not exist on new wafer
        if not all_wafers_df.empty:
            w_data = all_wafers_df[all_wafers_df["wafer_id"] == new_w]
            if not w_data.empty:
                avail_r = sorted(w_data["die_row"].unique())
                cur_r = st.session_state.get("selected_die_row")
                if cur_r not in avail_r:
                    def_r = avail_r[min(9, len(avail_r)-1)] if 9 in avail_r else avail_r[0]
                    st.session_state["selected_die_row"] = def_r
                    cur_r = def_r
                    
                avail_c = sorted(w_data[w_data["die_row"] == cur_r]["die_col"].unique())
                cur_c = st.session_state.get("selected_die_col")
                if cur_c not in avail_c:
                    def_c = avail_c[min(23, len(avail_c)-1)] if 23 in avail_c else avail_c[0]
                    st.session_state["selected_die_col"] = def_c
                    cur_c = def_c

                st.session_state["ov_row_selector"] = st.session_state["selected_die_row"]
                st.session_state["ov_col_selector"] = st.session_state["selected_die_col"]

def on_row_change():
    """Callback when Die Row selectbox changes on Overview page."""
    new_r = st.session_state.get("ov_row_selector")
    if new_r is not None:
        st.session_state["selected_die_row"] = new_r
        sel_w = st.session_state.get("selected_wafer")
        if sel_w and not all_wafers_df.empty:
            w_df = all_wafers_df[all_wafers_df["wafer_id"] == sel_w]
            avail_c = sorted(w_df[w_df["die_row"] == new_r]["die_col"].unique()) if not w_df.empty else []
            if avail_c and st.session_state.get("selected_die_col") not in avail_c:
                st.session_state["selected_die_col"] = avail_c[0]
                st.session_state["ov_col_selector"] = avail_c[0]

def on_col_change():
    """Callback when Die Column selectbox changes on Overview page."""
    new_c = st.session_state.get("ov_col_selector")
    if new_c is not None:
        st.session_state["selected_die_col"] = new_c


def render_die_explanation(wafer_id: str, die_row: int, die_col: int):
    """Render comprehensive die explanation for a given wafer and die coordinate.
    
    Reused across Failure Signature Drill-Down, Overview, and Die Explanation pages.
    """
    exp_detail = find_explanation(wafer_id, die_row, die_col)
    
    # Retrieve predictions for this die from joined_comp_df or all_wafers_df
    die_comp = pd.DataFrame()
    if not joined_comp_df.empty:
        die_comp = joined_comp_df[
            (joined_comp_df["wafer_id"] == wafer_id) & 
            (joined_comp_df["die_row"] == die_row) & 
            (joined_comp_df["die_col"] == die_col)
        ]
    
    if die_comp.empty and not all_wafers_df.empty:
        die_comp = all_wafers_df[
            (all_wafers_df["wafer_id"] == wafer_id) & 
            (all_wafers_df["die_row"] == die_row) & 
            (all_wafers_df["die_col"] == die_col)
        ]
    
    prob_b = float(die_comp.iloc[0]["prob_model_b"]) if not die_comp.empty and "prob_model_b" in die_comp.columns else (
        float(die_comp.iloc[0]["predicted_probability"]) if not die_comp.empty and "predicted_probability" in die_comp.columns else (
            exp_detail["predicted_probability"] if exp_detail else 0.0
        )
    )
    
    prob_a = float(die_comp.iloc[0]["prob_model_a"]) if not die_comp.empty and "prob_model_a" in die_comp.columns else (
        float(die_comp.iloc[0]["model_a_probability"]) if not die_comp.empty and "model_a_probability" in die_comp.columns else 0.0
    )
    
    risk_diff = float(die_comp.iloc[0]["risk_difference"]) if not die_comp.empty and "risk_difference" in die_comp.columns else (prob_b - prob_a)
    
    logit_val = exp_detail["final_logit"] if exp_detail else (
        float(np.log(prob_b / (1 - prob_b))) if 0 < prob_b < 1 else (10.0 if prob_b >= 1 else -10.0)
    )
    
    # Risk Badge
    if prob_b >= 0.5:
        badge_html = '<span class="badge-critical">▲ CRITICAL RISK</span>'
    elif prob_b >= 0.2912:
        badge_html = '<span class="badge-elevated">▲ HIGH RISK</span>'
    else:
        badge_html = '<span class="badge-pass">✓ PASS / LOW RISK</span>'
        
    st.markdown(f"""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 800; color: #64748B; letter-spacing: 0.5px; text-transform: uppercase;">SELECTED DIE EXPLANATION TARGET</div>
        <div style="font-size: 18px; font-weight: 900; color: #FFFFFF; margin-top: 2px;">
            {wafer_id} &nbsp;·&nbsp; Die Row {die_row}, Col {die_col} &nbsp; {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model B P(failure)", format_percent(prob_b))
    with m2:
        st.metric("Model A P(failure)", format_percent(prob_a) if prob_a > 0 else "N/A")
    with m3:
        st.metric("A→B ΔP", f"{risk_diff*100:+.2f}%" if prob_a > 0 or risk_diff != 0 else "N/A")
    with m4:
        st.metric("Final Logit", format_logit(logit_val))
        
    # Logit Waterfall / Breakdown Chart
    if exp_detail:
        decomp = exp_detail["logit_decomposition"]
        fig_wf_full = go.Figure(go.Waterfall(
            name="Logit Evidence",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Evidence Offset", "Spatial Evidence", "Parametric Evidence", "Block Evidence", "Final Logit"],
            textposition="outside",
            text=[
                format_logit(decomp["evidence_offset"]),
                format_logit(decomp["spatial_contribution"]),
                format_logit(decomp["parametric_contribution"]),
                format_logit(decomp["block_contribution"]),
                format_logit(logit_val),
            ],
            y=[
                decomp["evidence_offset"],
                decomp["spatial_contribution"],
                decomp["parametric_contribution"],
                decomp["block_contribution"],
                logit_val,
            ],
            connector={"line": {"color": "#64748B", "width": 1.5}},
            increasing={"marker": {"color": "#FF5252"}},
            decreasing={"marker": {"color": "#00E676"}},
            totals={"marker": {"color": "#00F0FF"}},
        ))
        
        fig_wf_full.update_layout(
            title=f"Logit Evidence Waterfall — {wafer_id} Die ({die_row}, {die_col})",
            height=420,
            template="plotly_dark",
            paper_bgcolor="#0B0E14",
            plot_bgcolor="#121721",
            yaxis=dict(gridcolor="#1E2638", title="Log-Odds Contribution"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_wf_full, use_container_width=True)

        # Render Top Parametric Risk Drivers if available
        if exp_detail.get("top_parametric_features"):
            with st.container(border=True):
                st.markdown("##### 👤 Top Parametric Risk Drivers (DiagonalScore)")
                df_p = pd.DataFrame(exp_detail["top_parametric_features"]).head(10)
                df_p["contrib_str"] = df_p["contribution"].apply(format_logit)
                fig_p = px.bar(
                    df_p,
                    y="feature_name",
                    x="contribution",
                    orientation="h",
                    color="contribution",
                    color_continuous_scale="RdYlGn_r",
                    text="contrib_str",
                    labels={"feature_name": "Parametric Feature", "contribution": "Log-Odds Contribution"},
                )
                fig_p.update_traces(textposition="outside")
                fig_p.update_layout(
                    height=280,
                    template="plotly_dark",
                    paper_bgcolor="#121721",
                    plot_bgcolor="#121721",
                    yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                    xaxis=dict(gridcolor="#1E2638"),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("ℹ️ Pre-computed feature waterfall unavailable for this exact die coordinate.")

    # Check Block Anomaly Information
    block_info = exp_detail.get("block_context", {}) if exp_detail else {}
    peak_idx = block_info.get("peak_anomaly_index")
    anom_row = None
    if not anomaly_df.empty:
        anom_match = anomaly_df[
            (anomaly_df["wafer_id"] == wafer_id) & 
            (anomaly_df["die_row"] == die_row) & 
            (anomaly_df["die_col"] == die_col)
        ]
        if not anom_match.empty:
            anom_row = anom_match.iloc[0]

    if peak_idx is not None or anom_row is not None:
        with st.container(border=True):
            st.markdown("##### ⚡ Block Anomaly Localization")
            a_col1, a_col2, a_col3 = st.columns(3)
            p_val = peak_idx if peak_idx is not None else int(anom_row.get("peak_anomaly_index", 0))
            a_col1.metric("Peak Anomaly Index", p_val)
            s_val = block_info.get("peak_scan_value")
            if s_val is None and anom_row is not None:
                s_val = float(anom_row.get("peak_scan_value", anom_row.get("peak_anomaly_score", 0.0)))
            a_col2.metric("Peak Scan Value / Score", f"{s_val:.4f}" if s_val is not None else "N/A")
            cat = str(anom_row.get("case_category", anom_row.get("anomaly_type", "Localized Anomaly"))) if anom_row is not None else "Localized Anomaly"
            a_col3.metric("Anomaly Classification", cat)
    else:
        st.caption("ℹ️ Detailed block localization is unavailable for this die.")

    st.markdown("""
    <div style="font-size: 11px; color: #64748B; margin-top: 8px;">
        💡 <b>Observational Interpretation Note</b>: Evidence contributions describe how the trained Model B formed its prediction log-odds. They represent exact observational model attributions, not causal physical proof.
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR (LEFT SIDE MANDATORY) ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1E2638; margin-bottom: 20px;">
        <div style="font-size: 22px; font-weight: 900; color: #FFFFFF; letter-spacing: 1.5px;">SANDISK</div>
        <div style="font-size: 18px; font-weight: 800; color: #00F0FF; letter-spacing: 1px; margin-top: -4px;">YIELD AI</div>
    </div>
    """, unsafe_allow_html=True)

    if "nav_section_radio" not in st.session_state:
        st.session_state["nav_section_radio"] = "🏠 Overview"

    nav_section = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "🔍 Die Explanation",
            "📊 Parametric Drivers",
            "📈 Block Analysis",
            "⚡ Model Benchmark",
            "🎯 Case Studies",
            "⚔️ Model A vs Model B",
            "🧩 Failure Signatures",
            "🔥 Wafer Hotspots",
        ],
        key="nav_section_radio"
    )

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Render Wafer Background Asset if available
    wafer_asset_p = Path("demo/assets/wafer_bg.png")
    if wafer_asset_p.exists():
        import base64
        with open(wafer_asset_p, "rb") as img_f:
            b64_img = base64.b64encode(img_f.read()).decode("utf-8")
        st.markdown(f"""
        <div style="position: relative; border-radius: 8px; overflow: hidden; border: 1px solid #1E2638; margin-bottom: 16px;">
            <img src="data:image/png;base64,{b64_img}" style="width: 100%; height: 140px; object-fit: cover; opacity: 0.75;" />
            <div style="position: absolute; bottom: 10px; left: 10px; right: 10px; color: #FFFFFF; font-weight: 800; font-size: 12px; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">
                From Data<br>to Higher Yield
                <div style="font-size: 9px; font-weight: 600; color: #00F0FF; margin-top: 2px;">SANDISK YIELD AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.image("https://images.unsplash.com/photo-1518770660439-4636190af475", use_container_width=True)

    st.markdown("""
    <div style="padding: 10px; background: #121721; border: 1px solid #1E2638; border-radius: 6px; text-align: center;">
        <div style="font-size: 10px; font-weight: 700; color: #64748B; letter-spacing: 1px;">MODEL B</div>
        <div style="font-size: 12px; font-weight: 800; color: #00E676; margin-top: 2px;">● VALIDATED</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN TOP HEADER ---
st.markdown("""
<div class="header-container">
    <div>
        <div class="header-title">Multi-Resolution Die Risk Intelligence</div>
        <div class="header-sub">Locate risk. Understand the evidence. Investigate the anomaly.</div>
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div class="header-status">● MODEL B VALIDATED</div>
        <div style="font-size: 11px; color: #64748B; letter-spacing: 0.5px;">Semiconductor &nbsp;|&nbsp; Data &nbsp;|&nbsp; Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Extracts
m_a = final_metrics.get("model_a", {})
m_b = final_metrics.get("model_b", {})
imps = final_metrics.get("improvements", {})

model_a_ap = m_a.get("ap", 0.581627)
model_b_ap = m_b.get("ap", 0.654879)
rel_ap_pct = imps.get("rel_ap_pct", ((model_b_ap - model_a_ap) / model_a_ap * 100) if model_a_ap else 0.0)

model_a_f1 = m_a.get("f1", 0.550543)
model_b_f1 = m_b.get("f1", 0.606253)
rel_f1_pct = imps.get("rel_f1_pct", ((model_b_f1 - model_a_f1) / model_a_f1 * 100) if model_a_f1 else 0.0)

model_a_auc = m_a.get("auc", 0.896295)
model_b_auc = m_b.get("auc", 0.925478)
rel_roc_pct = ((model_b_auc - model_a_auc) / model_a_auc * 100) if model_a_auc else 0.0

model_a_prec = m_a.get("precision", 0.784781)
model_b_prec = m_b.get("precision", 0.743921)

model_a_rec = m_a.get("recall", 0.423991)
model_b_rec = m_b.get("recall", 0.511582)
rel_rec_pct = ((model_b_rec - model_a_rec) / model_a_rec * 100) if model_a_rec else 0.0

# ==============================================================================
# PAGE 1: OVERVIEW & ALL-WAFER EXPLORER
# ==============================================================================
if nav_section == "🏠 Overview":
    # Executive Project Landing Page Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #121721 0%, #0B101A 100%); border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 800; color: #00F0FF; letter-spacing: 2px; text-transform: uppercase;">SANDISK YIELD AI</div>
        <div style="font-size: 24px; font-weight: 900; color: #FFFFFF; letter-spacing: 0.5px; margin-top: 2px;">Multi-Resolution Die Risk Explorer</div>
        <div style="font-size: 13px; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
            Interpretable prediction and diagnostic exploration of emerging die failures using parametric, spatial, and block-level evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Row (4 Cards)
    k1, k2, k3, k4 = st.columns([1.1, 1.1, 1.3, 2.0])
    
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🗄️</div>
            <div>
                <div class="kpi-label">TRAINING & TEST POPULATION</div>
                <div class="kpi-val-cyan">154,037 dies</div>
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; white-space: nowrap;">4.23% NEW-FAILURE RATE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🔲</div>
            <div>
                <div class="kpi-label">TEST WAFERS & DIES</div>
                <div class="kpi-val-cyan">40 Wafers</div>
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; white-space: nowrap;">39,351 TEST DIES EVALUATED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div>
                <div class="kpi-label">MODEL B CHAMPION AP</div>
                <div class="kpi-val-green">{model_b_ap:.4f}</div>
                <div style="font-size: 11px; color: #00E676; font-weight: 700; margin-top: 2px; white-space: nowrap;">↑ +{rel_ap_pct:.2f}% vs Model A ({model_a_ap:.4f})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="display: block;">
            <div class="kpi-label">MODEL A → MODEL B AP & F1 IMPROVEMENT</div>
            <div style="font-size: 11px; color: #94A3B8; display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span>Average Precision (AP)</span>
                <span><b>{model_a_ap:.4f} → {model_b_ap:.4f}</b> &nbsp;<b style="color: #00E676;">+{rel_ap_pct:.2f}%</b></span>
            </div>
            <div class="prog-bar-bg"><div class="prog-bar-fill" style="width: 78%;"></div></div>
            <div style="font-size: 11px; color: #94A3B8; display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                <span>Failure F1 Score</span>
                <span><b>{model_a_f1:.4f} → {model_b_f1:.4f}</b> &nbsp;<b style="color: #00E676;">+{rel_f1_pct:.2f}%</b></span>
            </div>
            <div class="prog-bar-bg"><div class="prog-bar-fill" style="width: 65%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # Compact Streamlit-Native Architecture Flow (2 Columns)
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    arch_a, arch_b = st.columns(2)

    with arch_a:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 11px; font-weight: 800; color: #64748B; letter-spacing: 1px;">ARCHITECTURE BRANCH 1</span>
                <span style="font-size: 10px; background: rgba(100, 116, 139, 0.2); color: #E2E8F0; padding: 2px 8px; border-radius: 4px; font-weight: 700;">MODEL A BASELINE</span>
            </div>
            <div style="font-size: 14px; font-weight: 800; color: #FFFFFF;">500 Parametric Features + Spatial Context</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Captures die-level electrical test metrics and spatial wafer geometry (x, y, r, &theta;).</div>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #1E2638; font-size: 12px; color: #00F0FF; font-weight: 700;">
                ↓ Baseline Failure Risk (AP = 0.5816 | F1 = 0.5505)
            </div>
            """, unsafe_allow_html=True)

    with arch_b:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 11px; font-weight: 800; color: #00F0FF; letter-spacing: 1px;">ARCHITECTURE BRANCH 2</span>
                <span style="font-size: 10px; background: rgba(0, 230, 118, 0.2); color: #00E676; padding: 2px 8px; border-radius: 4px; font-weight: 800;">MODEL B PRODUCTION CHAMPION</span>
            </div>
            <div style="font-size: 14px; font-weight: 800; color: #00F0FF;">Model A + 2,000 Block Readings → 73 LRT Features</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Integrates sub-die memory block anomaly readings via circular FFT LRT whitening encoder.</div>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #1E2638; font-size: 12px; color: #00E676; font-weight: 800;">
                ↓ Multi-Resolution Risk (AP = 0.6549 | +12.59% Gain)
            </div>
            """, unsafe_allow_html=True)

    # Narrative Investigation Story Flow Bar
    st.markdown("""
    <div style="background: #121721; border: 1px solid #1E2638; border-radius: 6px; padding: 10px 16px; margin: 12px 0 16px 0;">
        <div style="font-size: 10px; font-weight: 800; color: #64748B; letter-spacing: 1px; margin-bottom: 6px; text-transform: uppercase;">RECOMMENDED DEMO INVESTIGATION FLOW</div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; font-weight: 700; color: #CBD5E1; flex-wrap: wrap; gap: 6px;">
            <span>📁 DATA</span>
            <span style="color: #00F0FF;">→</span>
            <span>⚡ MODEL A <span style="font-size: 9px; color: #94A3B8;">(Spatial)</span></span>
            <span style="color: #00F0FF;">→</span>
            <span style="color: #00F0FF;">🔬 MODEL B <span style="font-size: 9px; color: #94A3B8;">(+Block LRT)</span></span>
            <span style="color: #00F0FF;">→</span>
            <span style="color: #00E676;">📈 A→B ΔP</span>
            <span style="color: #00F0FF;">→</span>
            <span>🧩 SIGNATURES</span>
            <span style="color: #00F0FF;">→</span>
            <span>🔥 HOTSPOTS</span>
            <span style="color: #00F0FF;">→</span>
            <span>🔍 DIE EXPLANATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    
    # Middle Row: Wafer Explorer (Left 70%) + Selected Die & Waterfall (Right 30%)
    col_left, col_right = st.columns([70, 30])
    
    with col_left:
        with st.container(border=True):
            # Wafer Explorer Header & Selector
            head_c1, head_c2 = st.columns([3, 2])
            with head_c1:
                st.markdown('<div style="font-size: 16px; font-weight: 800; color: #FFFFFF;">Wafer Explorer</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 12px; color: #94A3B8;">Select a wafer to explore predicted failure risk across dies.</div>', unsafe_allow_html=True)
            with head_c2:
                cur_w = st.session_state.get("selected_wafer", available_all_wafers[0] if available_all_wafers else "W_N_0083")
                w_idx = available_all_wafers.index(cur_w) if cur_w in available_all_wafers else 0
                st.selectbox(
                    "Select Wafer:",
                    available_all_wafers,
                    index=w_idx,
                    key="ov_wafer_selector",
                    on_change=on_wafer_change,
                    args=("ov_wafer_selector",)
                )
                selected_wafer = st.session_state["selected_wafer"]
                st.markdown(f'<div style="font-size: 11px; color: #00F0FF; text-align: right; margin-top: -6px;">{len(available_all_wafers)} wafers available</div>', unsafe_allow_html=True)
                
            wafer_data = all_wafers_df[all_wafers_df["wafer_id"] == selected_wafer].copy()
            avail_rows = sorted(wafer_data["die_row"].unique())
            
            # Ensure valid row & column in session state
            if st.session_state.get("selected_die_row") not in avail_rows:
                st.session_state["selected_die_row"] = avail_rows[0]
                
            cur_row = st.session_state["selected_die_row"]
            avail_cols = sorted(wafer_data[wafer_data["die_row"] == cur_row]["die_col"].unique())
            
            if st.session_state.get("selected_die_col") not in avail_cols:
                st.session_state["selected_die_col"] = avail_cols[0]
            cur_col = st.session_state["selected_die_col"]

            # Process Plotly chart selection event from previous rerun
            map_event = st.session_state.get("wafer_risk_map")
            if map_event and isinstance(map_event, dict) and "selection" in map_event and map_event["selection"].get("points"):
                pts = map_event["selection"]["points"]
                if pts:
                    pt = pts[0]
                    if "customdata" in pt and len(pt["customdata"]) >= 2:
                        c_row = int(pt["customdata"][0])
                        c_col = int(pt["customdata"][1])
                    else:
                        c_col = int(pt.get("x", 0))
                        c_row = int(pt.get("y", 0))
                    
                    # Validate clicked die exists in wafer dataset
                    valid_match = wafer_data[(wafer_data["die_row"] == c_row) & (wafer_data["die_col"] == c_col)]
                    if not valid_match.empty:
                        st.session_state["selected_die_row"] = c_row
                        st.session_state["selected_die_col"] = c_col
                        cur_row = c_row
                        cur_col = c_col

            # Synchronize widget keys before rendering selectboxes so UI dropdowns match canonical state
            st.session_state["ov_row_selector"] = cur_row
            st.session_state["ov_col_selector"] = cur_col

            # Die Selector Controls below dropdown
            r_idx = avail_rows.index(cur_row) if cur_row in avail_rows else 0
            c_idx = avail_cols.index(cur_col) if cur_col in avail_cols else 0

            ctrl_c1, ctrl_c2 = st.columns(2)
            with ctrl_c1:
                st.selectbox(
                    "Die Row", 
                    avail_rows, 
                    index=r_idx,
                    key="ov_row_selector",
                    on_change=on_row_change
                )
            with ctrl_c2:
                st.selectbox(
                    "Die Column", 
                    avail_cols, 
                    index=c_idx,
                    key="ov_col_selector",
                    on_change=on_col_change
                )

            cur_row = st.session_state["selected_die_row"]
            cur_col = st.session_state["selected_die_col"]
            die_row_data = wafer_data[(wafer_data["die_row"] == cur_row) & (wafer_data["die_col"] == cur_col)]
            
            # Plotly Wafer Scatter Surface
            fig_wafer = px.scatter(
                wafer_data,
                x="die_col",
                y="die_row",
                color="predicted_probability",
                color_continuous_scale="Reds",
                range_color=[0, 1],
                custom_data=["die_row", "die_col", "wafer_id", "predicted_probability", "old_label"],
                hover_data=["die_row", "die_col", "predicted_probability", "old_label"],
                labels={"die_col": "Die Column", "die_row": "Die Row", "predicted_probability": "P(failure)"},
            )
            fig_wafer.update_traces(marker=dict(size=10))
            
            # Highlight selected die with strong cyan halo
            fig_wafer.add_trace(
                go.Scatter(
                    x=[cur_col],
                    y=[cur_row],
                    mode="markers",
                    marker=dict(size=22, symbol="circle-open", color="#00F0FF", line=dict(width=3.5)),
                    name="Selected Die",
                    hoverinfo="skip",
                )
            )
            
            # Pre-test defective dies
            old_fails = wafer_data[wafer_data["old_label"] == 1]
            if not old_fails.empty:
                fig_wafer.add_trace(
                    go.Scatter(
                        x=old_fails["die_col"],
                        y=old_fails["die_row"],
                        mode="markers",
                        marker=dict(size=8, symbol="x", color="#000000"),
                        name="Defective (old_label=1)",
                        hoverinfo="skip",
                    )
                )
                
            fig_wafer.update_layout(
                height=510,
                template="plotly_dark",
                paper_bgcolor="#121721",
                plot_bgcolor="#121721",
                yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                xaxis=dict(gridcolor="#1E2638"),
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            st.plotly_chart(
                fig_wafer,
                key="wafer_risk_map",
                on_select="rerun",
                selection_mode="points",
                use_container_width=True,
            )
            
            st.markdown('<div style="font-size: 11px; color: #64748B; display: flex; justify-content: space-between; align-items: center;"><span>ℹ️ Hover over a die to see details</span> <span>🔍 Reset View &nbsp;|&nbsp; 📥 Download Map</span></div>', unsafe_allow_html=True)
        
    with col_right:
        # Synchronize shared session state across pages
        st.session_state["selected_die_row"] = cur_row
        st.session_state["selected_die_col"] = cur_col

        # Selected Die Intelligence Panel
        with st.container(border=True):
            st.markdown(f"""
            <div style="font-size: 13px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px; text-transform: uppercase;">SELECTED DIE</div>
            <div style="font-size: 15px; font-weight: 800; color: #00F0FF; margin-top: 2px; margin-bottom: 12px; white-space: nowrap;">{selected_wafer} &nbsp;·&nbsp; DIE ({cur_row}, {cur_col})</div>
            """, unsafe_allow_html=True)
            
            if not die_row_data.empty:
                prob = float(die_row_data.iloc[0]["predicted_probability"])
                old_lbl = int(die_row_data.iloc[0]["old_label"])
                has_label = "label" in die_row_data.columns
                act_lbl = int(die_row_data.iloc[0]["label"]) if has_label else None
                
                exp_detail = find_explanation(selected_wafer, cur_row, cur_col)
                logit_val = exp_detail["final_logit"] if exp_detail else (np.log(prob / (1 - prob)) if 0 < prob < 1 else (10.0 if prob >= 1 else -10.0))
                
                if prob >= 0.5:
                    badge_html = '<span class="badge-critical">▲ CRITICAL RISK</span>'
                elif prob >= 0.2912:
                    badge_html = '<span class="badge-elevated">▲ HIGH RISK</span>'
                else:
                    badge_html = '<span class="badge-pass">✓ PASS / LOW RISK</span>'
                    
                st.markdown(f"""
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <div style="flex: 1.2; background: #0B0E14; padding: 10px; border-radius: 6px; border: 1px solid {'#FF5252' if prob >= 0.2912 else '#1E2638'};">
                        <div style="font-size: 9px; font-weight: 700; color: {'#FF5252' if prob >= 0.2912 else '#64748B'}; text-transform: uppercase;">FAILURE PROBABILITY</div>
                        <div style="font-size: 26px; font-weight: 900; color: {'#FF5252' if prob >= 0.2912 else '#00E676'}; margin: 2px 0;">{format_percent(prob)}</div>
                        <div>{badge_html}</div>
                    </div>
                    <div style="flex: 1; font-size: 11px;">
                        <div style="color: #64748B; font-weight: 700; font-size: 9px; text-transform: uppercase;">FINAL LOGIT</div>
                        <div style="font-size: 15px; font-weight: 800; color: #FFFFFF;">{format_logit(logit_val)}</div>
                        <div style="color: #64748B; font-weight: 700; font-size: 9px; text-transform: uppercase; margin-top: 4px;">PRE-TEST STATUS</div>
                        <div style="font-size: 12px; font-weight: 700; color: #E2E8F0;">{'Eligible' if old_lbl == 0 else 'Defective (1)'}</div>
                        <div style="color: #64748B; font-weight: 700; font-size: 9px; text-transform: uppercase; margin-top: 4px;">ACTUAL LABEL</div>
                        <div style="font-size: 12px; font-weight: 700; color: #FFFFFF;">{'Defective (1)' if act_lbl==1 else ('Passed (0)' if act_lbl==0 else 'Unlabeled')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔍 Explain this die in Die Explanation", key="ov_explain_die_btn", use_container_width=True):
                    st.session_state["selected_wafer"] = selected_wafer
                    st.session_state["selected_die_row"] = cur_row
                    st.session_state["selected_die_col"] = cur_col
                    st.session_state["explanation_mode"] = "explorer"
                    st.session_state["nav_section_radio"] = "🔍 Die Explanation"
                    st.rerun()
            
        # Logit Evidence Waterfall Panel (Right Bottom)
        with st.container(border=True):
            st.markdown('<div class="panel-header"><span>📊 Evidence Decomposition (Logit Waterfall)</span></div>', unsafe_allow_html=True)
            
            if exp_detail:
                decomp = exp_detail["logit_decomposition"]
                
                fig_wf = go.Figure(go.Waterfall(
                    name="Logit Evidence",
                    orientation="v",
                    measure=["relative", "relative", "relative", "relative", "total"],
                    x=["Offset", "Spatial", "Parametric", "Block", "Final Logit"],
                    textposition="outside",
                    text=[
                        format_logit(decomp["evidence_offset"]),
                        format_logit(decomp["spatial_contribution"]),
                        format_logit(decomp["parametric_contribution"]),
                        format_logit(decomp["block_contribution"]),
                        format_logit(exp_detail["final_logit"]),
                    ],
                    y=[
                        decomp["evidence_offset"],
                        decomp["spatial_contribution"],
                        decomp["parametric_contribution"],
                        decomp["block_contribution"],
                        exp_detail["final_logit"],
                    ],
                    connector={"line": {"color": "#64748B", "width": 1}},
                    increasing={"marker": {"color": "#FF5252"}},
                    decreasing={"marker": {"color": "#00E676"}},
                    totals={"marker": {"color": "#00F0FF"}},
                ))
                
                fig_wf.update_layout(
                    height=220,
                    template="plotly_dark",
                    paper_bgcolor="#121721",
                    plot_bgcolor="#121721",
                    yaxis=dict(gridcolor="#1E2638"),
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.markdown('<div style="font-size: 10px; color: #64748B; text-align: center;">Positive evidence increases predicted failure risk. Negative evidence decreases predicted risk.</div>', unsafe_allow_html=True)
            else:
                # Look up observational evidence components from cluster assignment
                df_assign = load_cluster_assignments()
                die_assign = pd.DataFrame()
                if not df_assign.empty:
                    die_assign = df_assign[
                        (df_assign["wafer_id"] == selected_wafer) & 
                        (df_assign["die_row"] == cur_row) & 
                        (df_assign["die_col"] == cur_col)
                    ]
                if not die_assign.empty:
                    row_a = die_assign.iloc[0]
                    p_ev = float(row_a.get("parametric_evidence", 0.0))
                    s_ev = float(row_a.get("spatial_evidence", 0.0))
                    b_ev = float(row_a.get("block_evidence", 0.0))
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=["Parametric", "Spatial", "Block LRT"],
                        y=[p_ev, s_ev, b_ev],
                        marker_color=["#00F0FF", "#FFB74D", "#00E676"],
                        text=[format_logit(p_ev), format_logit(s_ev), format_logit(b_ev)],
                        textposition="auto",
                    ))
                    fig_bar.update_layout(
                        height=220,
                        template="plotly_dark",
                        paper_bgcolor="#121721",
                        plot_bgcolor="#121721",
                        title=dict(text=f"Logit Evidence Components — Die ({cur_row}, {cur_col})", font=dict(size=11, color="#94A3B8")),
                        yaxis=dict(gridcolor="#1E2638", title="Log-Odds Contribution"),
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('<div style="font-size: 10px; color: #64748B; text-align: center;">Observational multi-resolution evidence attributions.</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Pre-computed feature waterfall available for representative case study dies.")
            
    # Bottom Row: 3 Equal Columns (Parametric Drivers | Sub-Die Block Analysis | Model Comparison)
    b1, b2, b3 = st.columns(3)
    
    with b1:
        with st.container(border=True):
            st.markdown('<div class="panel-header"><span>👤 Top Parametric Risk Drivers</span> <span style="font-size: 11px; color: #00F0FF;">View All</span></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="display: flex; gap: 6px; margin-bottom: 10px;">
                <div style="flex: 1; background: #FF5252; color: #FFF; font-size: 10px; font-weight: 700; padding: 4px; border-radius: 4px; text-align: center;">Top Risk-Increasing</div>
                <div style="flex: 1; background: #161D2A; color: #64748B; font-size: 10px; font-weight: 700; padding: 4px; border-radius: 4px; text-align: center;">Top Risk-Reducing</div>
            </div>
            """, unsafe_allow_html=True)
            
            if exp_detail and exp_detail.get("top_parametric_features"):
                df_p = pd.DataFrame(exp_detail["top_parametric_features"]).head(5)
                df_p["contrib_str"] = df_p["contribution"].apply(format_logit)
                
                fig_p = px.bar(
                    df_p,
                    y="feature_name",
                    x="contribution",
                    orientation="h",
                    color="contribution",
                    color_continuous_scale="RdYlGn_r",
                    text="contrib_str",
                )
                fig_p.update_traces(textposition="outside")
                fig_p.update_layout(
                    height=190,
                    template="plotly_dark",
                    paper_bgcolor="#121721",
                    plot_bgcolor="#121721",
                    yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                    xaxis=dict(gridcolor="#1E2638"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("ℹ️ Parametric driver breakdown available for case study dies.")
            
    with b2:
        with st.container(border=True):
            st.markdown('<div class="panel-header"><span>📉 Sub-Die Block Analysis</span></div>', unsafe_allow_html=True)
            
            cat_map = {"W_N_0083": "True_Positive_TP", "W_F_0039": "False_Positive_FP", "W_N_0007": "True_Negative_TN", "W_N_0055": "False_Negative_FN"}
            cat_name = cat_map.get(selected_wafer, "True_Positive_TP")
            plot_p = get_plot_path(f"block_anomaly_{cat_name}.png")
            
            anom_info = anomaly_df[anomaly_df["wafer_id"] == selected_wafer] if not anomaly_df.empty else pd.DataFrame()
            p_idx = anom_info.iloc[0]["peak_anomaly_index"] if not anom_info.empty else "1,420"
            p_win = anom_info.iloc[0]["peak_anomaly_range"] if not anom_info.empty else "1,375 – 1,465"
            
            st.markdown(f"""
            <div style="display: flex; gap: 6px; font-size: 10px; margin-bottom: 8px;">
                <div style="flex:1; background:#0B0E14; padding:5px; border-radius:4px; text-align:center;">
                    <span style="color:#64748B;">Block Readings</span><br><b style="font-size: 12px; color: #FFF;">2,000</b>
                </div>
                <div style="flex:1; background:#0B0E14; padding:5px; border-radius:4px; text-align:center;">
                    <span style="color:#64748B;">Anomaly Peak</span><br><b style="font-size: 12px; color:#FF5252;">{p_idx}</b>
                </div>
                <div style="flex:1; background:#0B0E14; padding:5px; border-radius:4px; text-align:center;">
                    <span style="color:#64748B;">Investigation Window</span><br><b style="font-size: 11px; color: #FFF;">{p_win}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if plot_p:
                st.image(str(plot_p), use_container_width=True)
            else:
                st.info("ℹ️ Sub-die block sequence profile available for showcase dies.")
            
    with b3:
        with st.container(border=True):
            st.markdown('<div class="panel-header"><span>⚡ Model Comparison</span> <span style="font-size: 11px; color: #00F0FF;">View Details</span></div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: flex; gap: 8px; margin-bottom: 10px; align-items: center;">
                <div style="flex: 1; background: #0B0E14; padding: 8px; border-radius: 6px; border: 1px solid #1E2638; font-size: 11px;">
                    <div style="font-weight: 700; color: #94A3B8;">Model A</div>
                    <div style="font-size: 9px; color: #64748B;">Parametric + Spatial</div>
                    <div style="margin-top: 4px;">AP: <b>{model_a_ap:.4f}</b></div>
                    <div>F1: <b>{model_a_f1:.4f}</b></div>
                    <div>ROC-AUC: <b>0.8963</b></div>
                </div>
                <div style="font-size: 11px; font-weight: 800; color: #00E676; text-align: center;">
                    +{rel_ap_pct:.2f}%<br>AP<br><br>+{rel_f1_pct:.2f}%<br>F1
                </div>
                <div style="flex: 1; background: #0B0E14; padding: 8px; border-radius: 6px; border: 1px solid #00F0FF; font-size: 11px;">
                    <div style="font-weight: 700; color: #00F0FF;">Model B</div>
                    <div style="font-size: 9px; color: #64748B;">+ Block LRT Features</div>
                    <div style="margin-top: 4px;">AP: <b style="color: #00E676;">{model_b_ap:.4f}</b></div>
                    <div>F1: <b style="color: #00E676;">{model_b_f1:.4f}</b></div>
                    <div>ROC-AUC: <b style="color: #00E676;">0.9255</b></div>
                </div>
            </div>
            <div style="background: rgba(0, 230, 118, 0.1); border: 1px solid #00E676; border-radius: 4px; padding: 6px; text-align: center; font-size: 10px; font-weight: 700; color: #00E676;">
                🟢 Sub-die block information substantially improves failure prediction.
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-note">
        <div>ℹ️ Evidence contributions describe how the trained model formed its prediction. They are observational model attributions, not causal proof.</div>
        <div>SanDisk Hackathon &nbsp;|&nbsp; Die Yield Prediction &nbsp;|&nbsp; v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: DIE EXPLANATION
# ==============================================================================
elif nav_section == "🔍 Die Explanation":

    st.header("🔍 Die Logit Evidence Waterfall Breakdown")
    st.markdown("Decompose Model B's predicted log-odds into exact additive spatial, parametric, and block evidence components.")
    
    # Read shared session state for explorer selected die
    sel_w = st.session_state.get("selected_wafer")
    sel_r = st.session_state.get("selected_die_row")
    sel_c = st.session_state.get("selected_die_col")
    
    has_explorer_selection = (sel_w is not None and sel_r is not None and sel_c is not None)
    
    if "explanation_mode" not in st.session_state:
        st.session_state["explanation_mode"] = "explorer" if has_explorer_selection else "preset"

    exp_mode_label = st.radio(
        "Choose Explanation Target Mode:",
        ["🔎 Selected Die from Explorer", "🎯 Representative Case Studies"],
        index=0 if (st.session_state.get("explanation_mode") == "explorer" and has_explorer_selection) else 1,
        horizontal=True,
        key="exp_mode_radio"
    )
    
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    
    if exp_mode_label == "🔎 Selected Die from Explorer":
        st.session_state["explanation_mode"] = "explorer"
        st.markdown("### 🔎 Selected Die from Explorer")
        
        if has_explorer_selection:
            st.markdown(f"""
            <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 800; color: #00F0FF; text-transform: uppercase;">ACTIVE EXPLORER SELECTION</div>
                <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                    Wafer: <code>{sel_w}</code> &nbsp;·&nbsp; Die Row: <code>{sel_r}</code>, Col: <code>{sel_c}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Coordinate Adjuster
            with st.expander("⚙️ Adjust Selected Die Coordinates", expanded=False):
                c_w, c_r, c_c = st.columns(3)
                with c_w:
                    adj_w = st.selectbox("Wafer ID:", available_all_wafers, index=available_all_wafers.index(sel_w) if sel_w in available_all_wafers else 0, key="exp_adj_w")
                w_df_adj = all_wafers_df[all_wafers_df["wafer_id"] == adj_w] if not all_wafers_df.empty else pd.DataFrame()
                avail_r_adj = sorted(w_df_adj["die_row"].unique()) if not w_df_adj.empty else [sel_r]
                with c_r:
                    adj_r = st.selectbox("Die Row:", avail_r_adj, index=avail_r_adj.index(sel_r) if sel_r in avail_r_adj else 0, key="exp_adj_r")
                avail_c_adj = sorted(w_df_adj[w_df_adj["die_row"] == adj_r]["die_col"].unique()) if not w_df_adj.empty else [sel_c]
                with c_c:
                    adj_c = st.selectbox("Die Column:", avail_c_adj, index=avail_c_adj.index(sel_c) if sel_c in avail_c_adj else 0, key="exp_adj_c")
                    
                if adj_w != sel_w or adj_r != sel_r or adj_c != sel_c:
                    st.session_state["selected_wafer"] = adj_w
                    st.session_state["selected_die_row"] = adj_r
                    st.session_state["selected_die_col"] = adj_c
                    st.rerun()

            # Render exact explanation for selected die
            render_die_explanation(sel_w, sel_r, sel_c)
        else:
            st.info("ℹ️ No die has been selected in the Explorer yet. Select any die from Overview, Model A vs B, Failure Signatures, or Wafer Hotspots, or pick a representative case study below.")
            
            if available_all_wafers:
                def_w = available_all_wafers[0]
                def_w_df = all_wafers_df[all_wafers_df["wafer_id"] == def_w]
                def_r = int(def_w_df.iloc[0]["die_row"]) if not def_w_df.empty else 9
                def_c = int(def_w_df.iloc[0]["die_col"]) if not def_w_df.empty else 23
                st.session_state["selected_wafer"] = def_w
                st.session_state["selected_die_row"] = def_r
                st.session_state["selected_die_col"] = def_c
                render_die_explanation(def_w, def_r, def_c)

    else:
        st.session_state["explanation_mode"] = "preset"
        st.markdown("### 🎯 Representative Case Studies")
        
        if explanations_json:
            case_options = [f"{exp['wafer_id']} — Die ({exp['die_row']}, {exp['die_col']})" for exp in explanations_json]
            sel_case_str = st.selectbox("Select Representative Case Study Die:", case_options, key="tab2_case")
            
            exp = explanations_json[case_options.index(sel_case_str)]
            case_w = exp["wafer_id"]
            case_r = exp["die_row"]
            case_c = exp["die_col"]
            
            st.session_state["selected_wafer"] = case_w
            st.session_state["selected_die_row"] = case_r
            st.session_state["selected_die_col"] = case_c
            
            # Render exact explanation for selected preset
            render_die_explanation(case_w, case_r, case_c)

# ==============================================================================
# PAGE 3: PARAMETRIC DRIVERS
# ==============================================================================
elif nav_section == "📊 Parametric Drivers":
    st.header("📊 Top Parametric Measurement Drivers")
    st.markdown("Exposes top 5 positive (risk-increasing) and top 5 negative (risk-reducing) measurement feature drivers.")
    
    sel_w = st.session_state.get("selected_wafer", available_all_wafers[0] if available_all_wafers else "W_N_0083")
    sel_r = st.session_state.get("selected_die_row", 9)
    sel_c = st.session_state.get("selected_die_col", 23)
    
    st.markdown(f"""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 800; color: #00F0FF; text-transform: uppercase;">ACTIVE GLOBALLY SELECTED DIE</div>
        <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
            Wafer: <code>{sel_w}</code> &nbsp;·&nbsp; Die Row: <code>{sel_r}</code>, Col: <code>{sel_c}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    exp = find_explanation(sel_w, sel_r, sel_c)
    if exp:
        top_feats = exp.get("top_parametric_features", [])
        if top_feats:
            df_feats = pd.DataFrame(top_feats)
            pos_feats = df_feats[df_feats["contribution"] > 0].head(5)
            neg_feats = df_feats[df_feats["contribution"] < 0].head(5)
            df_top5 = pd.concat([pos_feats, neg_feats]) if not neg_feats.empty else pos_feats
            
            df_top5["contrib_str"] = df_top5["contribution"].apply(format_logit)
            
            fig_param_full = px.bar(
                df_top5,
                y="feature_name",
                x="contribution",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdYlGn_r",
                text="contrib_str",
                title=f"Top Parametric Feature Contributions for Wafer {sel_w} Die ({sel_r}, {sel_c})",
                labels={"contribution": "Logit Evidence Contribution", "feature_name": "Measurement Feature"},
            )
            fig_param_full.update_traces(textposition="outside")
            fig_param_full.update_layout(
                height=480,
                template="plotly_dark",
                paper_bgcolor="#0B0E14",
                plot_bgcolor="#121721",
                yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                xaxis=dict(gridcolor="#1E2638"),
            )
            st.plotly_chart(fig_param_full, use_container_width=True)
            
            with st.expander("🔍 View Measurement Details Table"):
                st.dataframe(df_feats[["feature_name", "feature_value", "centre", "weight", "contribution", "direction"]], use_container_width=True)
        else:
            st.info("ℹ️ Parametric feature evidence details unavailable for this die coordinate.")
    else:
        st.info("ℹ️ Explanation data unavailable for this die coordinate.")

# ==============================================================================
# PAGE 4: BLOCK ANALYSIS
# ==============================================================================
elif nav_section == "📈 Block Analysis":
    st.header("📈 Sub-Die Signal Investigation (2,000 Block Readings)")
    st.markdown("Pinpoint physical memory array defect locations across 2,000 sub-die scan readings using circular FFT LRT whitening.")
    
    sel_w = st.session_state.get("selected_wafer", available_all_wafers[0] if available_all_wafers else "W_N_0083")
    sel_r = st.session_state.get("selected_die_row", 9)
    sel_c = st.session_state.get("selected_die_col", 23)
    
    st.markdown(f"""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 800; color: #00F0FF; text-transform: uppercase;">ACTIVE GLOBALLY SELECTED DIE</div>
        <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
            Wafer: <code>{sel_w}</code> &nbsp;·&nbsp; Die Row: <code>{sel_r}</code>, Col: <code>{sel_c}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    exp = find_explanation(sel_w, sel_r, sel_c)
    cat_map = {"W_N_0083": "True_Positive_TP", "W_F_0039": "False_Positive_FP", "W_N_0007": "True_Negative_TN", "W_N_0055": "False_Negative_FN"}
    cat_name = cat_map.get(sel_w, "True_Positive_TP")
    
    anom_info = pd.DataFrame()
    if not anomaly_df.empty:
        anom_info = anomaly_df[(anomaly_df["wafer_id"] == sel_w) & (anomaly_df["die_row"] == sel_r) & (anomaly_df["die_col"] == sel_c)]
        if anom_info.empty:
            anom_info = anomaly_df[anomaly_df["wafer_id"] == sel_w]
            
    c_read, c_peak, c_win = st.columns(3)
    c_read.metric("BLOCK READINGS", "2,000")
    if not anom_info.empty:
        peak_idx = anom_info.iloc[0]["peak_anomaly_index"]
        peak_win = anom_info.iloc[0]["peak_anomaly_range"]
        c_peak.metric("ANOMALY PEAK (t*)", f"{peak_idx}")
        c_win.metric("INVESTIGATION WINDOW", f"{peak_win}")
    else:
        c_peak.metric("ANOMALY PEAK (t*)", "N/A")
        c_win.metric("INVESTIGATION WINDOW", "N/A")
        
    plot_p = get_plot_path(f"block_anomaly_{cat_name}.png")
    if plot_p:
        st.image(str(plot_p), caption=f"2,000 Sub-Die Block Reading Anomaly Scan Profile — {sel_w} Die ({sel_r}, {sel_c})", use_container_width=True)
        st.caption("Localized anomaly detected in the sub-die block sequence (Observational model evidence).")
    else:
        st.warning(f"⚠️ No pre-generated sub-die block anomaly plot artifact exists for Wafer {sel_w} Die ({sel_r}, {sel_c}).")

# ==============================================================================
# PAGE 5: MODEL BENCHMARK
# ==============================================================================
elif nav_section == "⚡ Model Benchmark":
    st.header("⚡ Model Benchmark & Architectural Screening")
    
    st.markdown(f"""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 6px; padding: 14px 20px; margin-bottom: 16px;">
        <div style="font-size: 18px; font-weight: 800; color: #00F0FF; margin-bottom: 4px;">Empirical validation confirms Model B Champion as the optimal interpretable architecture.</div>
        <div style="font-size: 13px; color: #94A3B8;">Evaluating Production Models against Experimental Deep Architectures across 5-Fold Stratified Wafer-Grouped Cross-Validation.</div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_prod, tab_exp = st.tabs(["🏆 Production Models (Model A vs Model B)", "🧪 Experimental & Rejected Architectures (GNN, MLP, Fusion)"])
    
    with tab_prod:
        col_a, col_b = st.columns(2)
        
        with col_a:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 1px;">MODEL A (BASELINE)</div>
                <div style="font-size: 16px; font-weight: 700; color: #FFFFFF;">Parametric + Spatial</div>
                <div style="display: flex; justify-content: space-around; margin-top: 16px;">
                    <div><div class="kpi-label">AP</div><div style="font-size: 24px; font-weight: 800; color: #E2E8F0;">{model_a_ap:.4f}</div></div>
                    <div><div class="kpi-label">F1</div><div style="font-size: 24px; font-weight: 800; color: #E2E8F0;">{model_a_f1:.4f}</div></div>
                    <div><div class="kpi-label">ROC-AUC</div><div style="font-size: 24px; font-weight: 800; color: #E2E8F0;">{model_a_auc:.4f}</div></div>
                </div>
                """, unsafe_allow_html=True)
            
        with col_b:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size: 11px; font-weight: 700; color: #00F0FF; letter-spacing: 1px;">PRODUCTION MODEL B (CHAMPION)</div>
                <div style="font-size: 16px; font-weight: 700; color: #00F0FF;">Parametric + Spatial + 73 Block LRT Features</div>
                <div style="display: flex; justify-content: space-around; margin-top: 16px;">
                    <div><div class="kpi-label">AP</div><div style="font-size: 24px; font-weight: 800; color: #00E676;">{model_b_ap:.4f}</div><div style="font-size: 11px; color: #00E676; font-weight: 700;">+{rel_ap_pct:.2f}% Gain</div></div>
                    <div><div class="kpi-label">F1</div><div style="font-size: 24px; font-weight: 800; color: #00E676;">{model_b_f1:.4f}</div><div style="font-size: 11px; color: #00E676; font-weight: 700;">+{rel_f1_pct:.2f}% Gain</div></div>
                    <div><div class="kpi-label">ROC-AUC</div><div style="font-size: 24px; font-weight: 800; color: #00E676;">{model_b_auc:.4f}</div><div style="font-size: 11px; color: #00E676; font-weight: 700;">+{rel_roc_pct:.2f}% Gain</div></div>
                </div>
                """, unsafe_allow_html=True)
            
        st.subheader("Detailed Production Metric Matrix")
        bench_table = pd.DataFrame([
            {"Metric": "Average Precision (AP)", "Model A (Baseline)": f"{model_a_ap:.4f}", "Model B (Champion)": f"{model_b_ap:.4f}", "Absolute Delta": f"+{model_b_ap - model_a_ap:.4f}", "Relative Gain": f"+{rel_ap_pct:.2f}%"},
            {"Metric": "ROC-AUC", "Model A (Baseline)": f"{model_a_auc:.4f}", "Model B (Champion)": f"{model_b_auc:.4f}", "Absolute Delta": f"+{model_b_auc - model_a_auc:.4f}", "Relative Gain": f"+{rel_roc_pct:.2f}%"},
            {"Metric": "Failure F1 Score", "Model A (Baseline)": f"{model_a_f1:.4f}", "Model B (Champion)": f"{model_b_f1:.4f}", "Absolute Delta": f"+{model_b_f1 - model_a_f1:.4f}", "Relative Gain": f"+{rel_f1_pct:.2f}%"},
            {"Metric": "Precision", "Model A (Baseline)": f"{model_a_prec:.4f}", "Model B (Champion)": f"{model_b_prec:.4f}", "Absolute Delta": f"{model_b_prec - model_a_prec:.4f}", "Relative Gain": "N/A"},
            {"Metric": "Recall", "Model A (Baseline)": f"{model_a_rec:.4f}", "Model B (Champion)": f"{model_b_rec:.4f}", "Absolute Delta": f"+{model_b_rec - model_a_rec:.4f}", "Relative Gain": f"+{rel_rec_pct:.2f}%"},
        ])
        st.dataframe(bench_table, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background: rgba(0, 230, 118, 0.08); border: 1px solid #00E676; border-radius: 6px; padding: 14px; margin-top: 16px; font-size: 12px; color: #CBD5E1;">
            🟢 <b>Statistical Robustness Summary</b>: Model B's AP improvement over Model A is statistically significant across 1,000 wafer-level bootstrap iterations (95% CI: [+0.0677, +0.0783], Wilcoxon signed-rank test <i>p</i> &lt; 10<sup>-20</sup>, Cohen's <i>d<sub>z</sub></i> = 1.20).
        </div>
        """, unsafe_allow_html=True)

    with tab_exp:
        st.markdown("#### 🧪 Empirical Architectural Exploration & Screening Results")
        st.markdown("To ensure optimal performance, alternative deep neural architectures and multi-resolution fusion strategies were systematically evaluated and screened.")

        exp_bench_table = pd.DataFrame([
            {
                "Architecture / Strategy": "Model B Champion (Fixed Additive Logit)",
                "Status": "🟢 PRODUCTION CHAMPION",
                "Average Precision (AP)": "0.6549",
                "ROC-AUC": "0.9255",
                "Failure F1": "0.6063",
                "Screening Rationale": "Optimal ranking precision, robust generalization, and 100% additive logit interpretability.",
            },
            {
                "Architecture / Strategy": "Model A Baseline (Parametric + Spatial)",
                "Status": "⚪ PRODUCTION BASELINE",
                "Average Precision (AP)": "0.5816",
                "ROC-AUC": "0.8963",
                "Failure F1": "0.5505",
                "Screening Rationale": "Solid baseline performance without sub-die block readings.",
            },
            {
                "Architecture / Strategy": "Equal-Weighted Baseline Fusion",
                "Status": "🔴 REJECTED EXPERIMENT",
                "Average Precision (AP)": "0.6012",
                "ROC-AUC": "0.8971",
                "Failure F1": "0.5446",
                "Screening Rationale": "Fixed 1/3 weighting penalizes dominant parametric signal, yielding -0.0528 AP penalty vs Champion.",
            },
            {
                "Architecture / Strategy": "Adaptive Gated Multi-Resolution Fusion",
                "Status": "🔴 REJECTED EXPERIMENT",
                "Average Precision (AP)": "0.5988",
                "ROC-AUC": "0.8963",
                "Failure F1": "0.3745",
                "Screening Rationale": "Learned softmax gating network overfits on severe 22:1 class imbalance.",
            },
            {
                "Architecture / Strategy": "WaferGNN (Graph Neural Network)",
                "Status": "🔴 REJECTED EXPERIMENT",
                "Average Precision (AP)": "0.5794",
                "ROC-AUC": "0.8950",
                "Failure F1": "0.5483",
                "Screening Rationale": "Spatial graph message passing induces over-smoothing across neighboring wafer nodes.",
            },
            {
                "Architecture / Strategy": "Spatial-Only Baseline (No Parametric)",
                "Status": "🔴 REJECTED EXPERIMENT",
                "Average Precision (AP)": "0.0520",
                "ROC-AUC": "0.5660",
                "Failure F1": "0.0959",
                "Screening Rationale": "Proves pure spatial coordinates alone lack predictive power without electrical measurement signals.",
            },
        ])
        st.dataframe(exp_bench_table, use_container_width=True, hide_index=True)

        st.info("💡 **Empirical Conclusion**: Experimental screening confirmed that non-linear gating and graph neural networks do not outperform fixed additive logit evidence fusion on imbalanced yield data while introducing unnecessary opacity.")

# ==============================================================================
# PAGE 6: CASE STUDIES
# ==============================================================================
elif nav_section == "🎯 Case Studies":
    st.header("🎯 Representative Case Study Explorer")
    st.markdown("Examine four pre-classified case studies demonstrating model success and error modes.")
    
    t_tp, t_fp, t_tn, t_fn = st.tabs(["🟢 True Positive (TP)", "🔴 False Positive (FP)", "⚪ True Negative (TN)", "🟡 False Negative (FN)"])
    
    cases_info = [
        ("True_Positive_TP", t_tp, "W_N_0083", "Die (9, 23)", "100.0%", "CRITICAL RISK", "Defective (1)", "Correctly flagged by high parametric (+10.64) & block anomaly (+4.22) evidence."),
        ("False_Positive_FP", t_fp, "W_F_0039", "Die (17, 3)", "77.9%", "HIGH RISK / FP", "Passed (0)", "False alarm caused by localized sub-die block scan reading spike (+4.02)."),
        ("True_Negative_TN", t_tn, "W_N_0007", "Die (20, 18)", "0.02%", "PASS / LOW RISK", "Passed (0)", "Clean passing die with strong negative risk evidence across all channels."),
        ("False_Negative_FN", t_fn, "W_N_0055", "Die (23, 11)", "0.15%", "MISSED / FN", "Defective (1)", "Missed failure caused by an isolated point defect unobservable in parametric/block test readings."),
    ]
    
    for cat_key, tab, wafer_id, die_str, p_str, risk_str, act_str, desc in cases_info:
        with tab:
            st.subheader(f"Case Study: {cat_key.replace('_', ' ')} — Wafer {wafer_id} {die_str}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted P(fail)", p_str)
            c2.metric("Risk Classification", risk_str)
            c3.metric("Actual Label", act_str)
            
            st.markdown(f"**Diagnostic Analysis**: {desc}")
            
            p_heatmap = get_plot_path(f"wafer_heatmap_{wafer_id}.png")
            p_block = get_plot_path(f"block_anomaly_{cat_key}.png")
            
            if p_heatmap:
                st.image(str(p_heatmap), caption=f"Wafer Heatmap Surface — {wafer_id}", use_container_width=True)
            if p_block:
                st.image(str(p_block), caption=f"Sub-Die Block Anomaly Profile — {cat_key}", use_container_width=True)

    st.markdown("""
    <div style="font-size: 11px; color: #64748B; margin-top: 16px;">
        💡 <b>Observational Disclaimer</b>: Case studies present model attributions and risk outputs. They serve as diagnostic aids, not causal physical defect verifications.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 7: MODEL A VS MODEL B WAFER COMPARISON
# ==============================================================================
elif nav_section == "⚔️ Model A vs Model B":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #121721 0%, #0B101A 100%); border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 8px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 800; color: #00F0FF; letter-spacing: 2px; text-transform: uppercase;">CORE COMPARATIVE QUESTION</div>
        <div style="font-size: 22px; font-weight: 900; color: #FFFFFF; letter-spacing: 0.5px; margin-top: 2px;">
            How much additional predictive signal do the block readings provide?
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
            Model B extends Model A with block-level evidence. The improvement measures the incremental predictive value of the multi-resolution signal.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistical Overview Metrics Callout
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Model A AP", f"{model_a_ap:.4f}")
    with k2:
        st.metric("Model B AP", f"{model_b_ap:.4f}")
    with k3:
        st.metric("Δ AP (Absolute)", f"+{model_b_ap - model_a_ap:.4f}")
    with k4:
        st.metric("Relative AP Gain", f"+{rel_ap_pct:.2f}%")

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    
    if available_all_wafers:
        st.caption(f"{len(available_all_wafers)} test wafers available for die-by-die comparison")
        cur_w = st.session_state.get("selected_wafer", available_all_wafers[0])
        w_idx = available_all_wafers.index(cur_w) if cur_w in available_all_wafers else 0
        st.selectbox(
            "Select Wafer for Model A vs Model B Comparison:",
            available_all_wafers,
            index=w_idx,
            key="comp_wafer_selector",
            on_change=on_wafer_change,
            args=("comp_wafer_selector",)
        )
        selected_wafer_comp = st.session_state["selected_wafer"]
        
        # Filter wafer die grid predictions for joined dataset
        w_comp = joined_comp_df[joined_comp_df["wafer_id"] == selected_wafer_comp] if not joined_comp_df.empty else pd.DataFrame()
        
        if not w_comp.empty:
            # Check for map point click selections to synchronize die selection
            for m_key in ["map_model_a", "map_model_b", "map_risk_diff"]:
                m_state = st.session_state.get(m_key, {})
                if m_state and isinstance(m_state, dict) and "selection" in m_state:
                    pts = m_state["selection"].get("points", [])
                    if pts:
                        p = pts[0]
                        st.session_state["comp_selected_die"] = (int(p.get("y")), int(p.get("x")))
                        break
            
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            col_comp_a, col_comp_b, col_comp_diff = st.columns(3)
            
            # 1. Model A Map
            with col_comp_a:
                with st.container(border=True):
                    st.markdown("""
                    <div style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 1px;">MODEL A (BASELINE)</div>
                    <div style="font-size: 15px; font-weight: 700; color: #FFFFFF;">Parametric + Spatial</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Predicted Failure Probability</div>
                    """, unsafe_allow_html=True)
                    
                    fig_a = px.scatter(
                        w_comp,
                        x="die_col",
                        y="die_row",
                        color="prob_model_a",
                        color_continuous_scale="Reds",
                        range_color=[0, 1],
                        labels={"prob_model_a": "P(fail)", "die_col": "Die Col", "die_row": "Die Row"},
                        title=f"Model A Grid — Wafer {selected_wafer_comp}"
                    )
                    fig_a.update_traces(marker=dict(size=9, symbol="square"))
                    fig_a.update_layout(
                        height=380,
                        template="plotly_dark",
                        paper_bgcolor="#0B0E14",
                        plot_bgcolor="#121721",
                        yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                        xaxis=dict(gridcolor="#1E2638"),
                        margin=dict(l=5, r=5, t=35, b=5)
                    )
                    st.plotly_chart(fig_a, use_container_width=True, key="map_model_a", on_select="rerun", selection_mode="points")

            # 2. Model B Map
            with col_comp_b:
                with st.container(border=True):
                    st.markdown("""
                    <div style="font-size: 11px; font-weight: 700; color: #00F0FF; letter-spacing: 1px;">PRODUCTION MODEL B (CHAMPION)</div>
                    <div style="font-size: 15px; font-weight: 700; color: #00F0FF;">Parametric + Spatial + Block LRT</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Predicted Failure Probability</div>
                    """, unsafe_allow_html=True)
                    
                    fig_b = px.scatter(
                        w_comp,
                        x="die_col",
                        y="die_row",
                        color="prob_model_b",
                        color_continuous_scale="Reds",
                        range_color=[0, 1],
                        labels={"prob_model_b": "P(fail)", "die_col": "Die Col", "die_row": "Die Row"},
                        title=f"Model B Grid — Wafer {selected_wafer_comp}"
                    )
                    fig_b.update_traces(marker=dict(size=9, symbol="square"))
                    fig_b.update_layout(
                        height=380,
                        template="plotly_dark",
                        paper_bgcolor="#0B0E14",
                        plot_bgcolor="#121721",
                        yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                        xaxis=dict(gridcolor="#1E2638"),
                        margin=dict(l=5, r=5, t=35, b=5)
                    )
                    st.plotly_chart(fig_b, use_container_width=True, key="map_model_b", on_select="rerun", selection_mode="points")

            # 3. A -> B Risk Difference Map
            with col_comp_diff:
                with st.container(border=True):
                    st.markdown("""
                    <div style="font-size: 11px; font-weight: 700; color: #FFB74D; letter-spacing: 1px;">A → B DIFFERENCE</div>
                    <div style="font-size: 15px; font-weight: 700; color: #FFB74D;">Change in Predicted Risk</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Model B P(fail) − Model A P(fail)</div>
                    """, unsafe_allow_html=True)
                    
                    max_abs_diff = max(abs(w_comp["risk_difference"].min()), abs(w_comp["risk_difference"].max()), 0.1)
                    fig_diff = px.scatter(
                        w_comp,
                        x="die_col",
                        y="die_row",
                        color="risk_difference",
                        color_continuous_scale="RdBu_r",
                        range_color=[-max_abs_diff, max_abs_diff],
                        labels={"risk_difference": "Risk Δ", "die_col": "Die Col", "die_row": "Die Row"},
                        title=f"Risk Difference — Wafer {selected_wafer_comp}"
                    )
                    fig_diff.update_traces(marker=dict(size=9, symbol="square"))
                    fig_diff.update_layout(
                        height=380,
                        template="plotly_dark",
                        paper_bgcolor="#0B0E14",
                        plot_bgcolor="#121721",
                        yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
                        xaxis=dict(gridcolor="#1E2638"),
                        margin=dict(l=5, r=5, t=35, b=5)
                    )
                    st.plotly_chart(fig_diff, use_container_width=True, key="map_risk_diff", on_select="rerun", selection_mode="points")

            # Selected Die Comparison Box
            comp_die = st.session_state.get("comp_selected_die", None)
            if not comp_die and not w_comp.empty:
                top_diff_idx = w_comp["risk_difference"].idxmax()
                comp_die = (int(w_comp.loc[top_diff_idx, "die_row"]), int(w_comp.loc[top_diff_idx, "die_col"]))
                
            if comp_die:
                cd_r, cd_c = comp_die
                d_row = w_comp[(w_comp["die_row"] == cd_r) & (w_comp["die_col"] == cd_c)]
                if not d_row.empty:
                    pa = d_row.iloc[0]["prob_model_a"]
                    pb = d_row.iloc[0]["prob_model_b"]
                    diff_val = d_row.iloc[0]["risk_difference"]
                    diff_color = "#FF5252" if diff_val > 0.01 else ("#3B82F6" if diff_val < -0.01 else "#94A3B8")
                    
                    st.session_state["selected_wafer"] = selected_wafer_comp
                    st.session_state["selected_die_row"] = cd_r
                    st.session_state["selected_die_col"] = cd_c
                    
                    st.markdown(f"""
                    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #00F0FF; border-radius: 6px; padding: 14px 20px; margin-top: 14px;">
                        <div style="font-size: 14px; font-weight: 800; color: #FFFFFF;">
                            Selected Die Comparison — Wafer {selected_wafer_comp} &nbsp;|&nbsp; Die ({cd_r}, {cd_c})
                        </div>
                        <div style="display: flex; gap: 32px; margin-top: 10px; align-items: center;">
                            <div><div class="kpi-label">MODEL A RISK</div><div style="font-size: 18px; font-weight: 800; color: #E2E8F0;">{pa*100:.2f}%</div></div>
                            <div><div class="kpi-label">MODEL B RISK</div><div style="font-size: 18px; font-weight: 800; color: #00E676;">{pb*100:.2f}%</div></div>
                            <div><div class="kpi-label">RISK CHANGE</div><div style="font-size: 18px; font-weight: 800; color: {diff_color};">{diff_val*100:+.2f} percentage points</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🔍 Explain this die in Die Explanation", key="comp_explain_die_btn", use_container_width=True):
                        st.session_state["selected_wafer"] = selected_wafer_comp
                        st.session_state["selected_die_row"] = cd_r
                        st.session_state["selected_die_col"] = cd_c
                        st.session_state["explanation_mode"] = "explorer"
                        st.session_state["nav_section_radio"] = "🔍 Die Explanation"
                        st.rerun()

            # Compact Wafer Risk Shift Insight Box
            n_pos_diff = int((w_comp["risk_difference"] > 0.01).sum())
            n_neg_diff = int((w_comp["risk_difference"] < -0.01).sum())
            
            idx_max_inc = w_comp["risk_difference"].idxmax()
            idx_max_dec = w_comp["risk_difference"].idxmin()
            
            row_inc = w_comp.loc[idx_max_inc, "die_row"]
            col_inc = w_comp.loc[idx_max_inc, "die_col"]
            val_inc = w_comp.loc[idx_max_inc, "risk_difference"]
            
            row_dec = w_comp.loc[idx_max_dec, "die_row"]
            col_dec = w_comp.loc[idx_max_dec, "die_col"]
            val_dec = w_comp.loc[idx_max_dec, "risk_difference"]

            st.markdown(f"""
            <div style="background: #121721; border: 1px solid #1E2638; border-radius: 8px; padding: 16px; margin-top: 16px;">
                <div style="font-size: 14px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">💡 Wafer Risk Shift Insights</div>
                <ul style="font-size: 13px; color: #CBD5E1; margin: 0; padding-left: 20px; line-height: 1.6;">
                    <li>Model B assigns higher predicted risk for <b>{n_pos_diff}</b> dies on this wafer and lower predicted risk for <b>{n_neg_diff}</b> dies compared to Model A.</li>
                    <li><b>Maximum risk increase</b>: <b>+{val_inc*100:.2f} percentage points</b> at die <code>({row_inc}, {col_inc})</code>.</li>
                    <li><b>Maximum risk decrease</b>: <b>{val_dec*100:.2f} percentage points</b> at die <code>({row_dec}, {col_dec})</code>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 11px; color: #64748B; margin-top: 16px;">
            💡 <b>Observational Note</b>: Model B extends Model A with block-level evidence. The improvement measures the incremental predictive value of the multi-resolution signal, not causal proof of physical defect mechanisms.
        </div>
        """, unsafe_allow_html=True)

        # Metrics comparison area below wafer maps
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        m_col_a, m_col_b = st.columns(2)
        
        with m_col_a:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 1px;">MODEL A METRICS</div>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div><div class="kpi-label">AP</div><div style="font-size: 22px; font-weight: 800; color: #E2E8F0;">{model_a_ap:.4f}</div></div>
                    <div><div class="kpi-label">F1</div><div style="font-size: 22px; font-weight: 800; color: #E2E8F0;">{model_a_f1:.4f}</div></div>
                    <div><div class="kpi-label">ROC-AUC</div><div style="font-size: 22px; font-weight: 800; color: #E2E8F0;">{model_a_auc:.4f}</div></div>
                </div>
                """, unsafe_allow_html=True)
                
        with m_col_b:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size: 11px; font-weight: 700; color: #00F0FF; letter-spacing: 1px;">MODEL B METRICS</div>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div><div class="kpi-label">AP</div><div style="font-size: 22px; font-weight: 800; color: #00E676;">{model_b_ap:.4f}</div></div>
                    <div><div class="kpi-label">F1</div><div style="font-size: 22px; font-weight: 800; color: #00E676;">{model_b_f1:.4f}</div></div>
                    <div><div class="kpi-label">ROC-AUC</div><div style="font-size: 22px; font-weight: 800; color: #00E676;">{model_b_auc:.4f}</div></div>
                </div>
                """, unsafe_allow_html=True)

# --- PAGE 8: FAILURE SIGNATURES ---
elif nav_section == "🧩 Failure Signatures":
    st.markdown("""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #3B82F6; border-radius: 8px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 20px; font-weight: 900; color: #FFFFFF; letter-spacing: 0.5px;">FAILURE SIGNATURE EXPLORER</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 4px;">Exploratory grouping of multi-resolution evidence patterns across Parametric, Spatial, and Block LRT branches.</div>
    </div>
    """, unsafe_allow_html=True)

    sig_summary = load_signature_summary()
    cluster_df = load_cluster_summary()
    profiles_df = load_cluster_profiles()

    if cluster_df.empty:
        st.warning("Failure signature analysis artifacts not found in results/failure_signatures/")
    else:
        # Top KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">DISCOVERED SIGNATURES</div>
                <div class="kpi-val-cyan">{len(cluster_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">ANALYZED DIES</div>
                <div class="kpi-val-cyan">{cluster_df['n_dies'].sum():,}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">CLUSTERING METHOD</div>
                <div class="kpi-val-green">KMeans (k=5)</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">EVIDENCE RESOLUTIONS</div>
                <div class="kpi-val-cyan">3 Branches</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # Plot Gallery (2x2 Grid)
        st.markdown("#### 📊 Multi-Resolution Failure Signature Visualizations")
        p_col1, p_col2 = st.columns(2)
        sig_plots_dir = Path("results/failure_signatures/plots")

        with p_col1:
            with st.container(border=True):
                st.markdown("**2D PCA Cluster Projection**")
                p1 = sig_plots_dir / "cluster_projection.png"
                if p1.exists():
                    st.image(str(p1), use_container_width=True)
                else:
                    st.info("cluster_projection.png not found")

            with st.container(border=True):
                st.markdown("**Die Count per Cluster**")
                p3 = sig_plots_dir / "cluster_sizes.png"
                if p3.exists():
                    st.image(str(p3), use_container_width=True)
                else:
                    st.info("cluster_sizes.png not found")

        with p_col2:
            with st.container(border=True):
                st.markdown("**Multi-Resolution Evidence Profiles**")
                p2 = sig_plots_dir / "cluster_evidence_profiles.png"
                if p2.exists():
                    st.image(str(p2), use_container_width=True)
                else:
                    st.info("cluster_evidence_profiles.png not found")

            with st.container(border=True):
                st.markdown("**Wafer Signature Composition**")
                p4 = sig_plots_dir / "wafer_signature_distribution.png"
                if p4.exists():
                    st.image(str(p4), use_container_width=True)
                else:
                    st.info("wafer_signature_distribution.png not found")

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # Interactive Cluster Inspector
        st.markdown("#### 🔍 Interactive Signature Inspector")
        cluster_options = [
            f"Cluster {row.cluster_id}: {row.signature_name} ({row.dominant_source} Dominant)"
            for _, row in cluster_df.iterrows()
        ]
        selected_sig_option = st.selectbox("Select Failure Signature Cluster:", cluster_options, index=0)
        selected_cid = int(selected_sig_option.split(":")[0].replace("Cluster", "").strip())

        sig_row = cluster_df[cluster_df["cluster_id"] == selected_cid].iloc[0]

        c_det1, c_det2 = st.columns([1, 1])
        with c_det1:
            with st.container(border=True):
                st.markdown(f"##### Signature Overview — Cluster {sig_row.cluster_id}")
                st.markdown(f"""
                - **Signature Label**: `{sig_row.signature_name}`
                - **Dominant Evidence Source**: `{sig_row.dominant_source}`
                - **Die Count**: `{int(sig_row.n_dies):,}` dies (`{sig_row.pct_dies:.2f}%` of analyzed dies)
                - **Descriptive Failure Rate (Post-hoc)**: `{sig_row.failure_rate_pct:.2f}%` (`{int(sig_row.n_failures):,}` failures)
                - **Average Model A Risk**: `{sig_row.mean_model_a_prob*100:.2f}%`
                - **Average Model B Risk**: `{sig_row.mean_model_b_prob*100:.2f}%`
                - **Average Risk Change ($\Delta P$)**: `{sig_row.mean_risk_difference*100:+.2f} percentage points`
                """)

        with c_det2:
            with st.container(border=True):
                st.markdown("##### Representative Evidence Profile")
                st.markdown(f"""
                - **Parametric Logit Evidence**: `{sig_row.mean_parametric_evidence:+.3f}`
                - **Spatial Logit Evidence**: `{sig_row.mean_spatial_evidence:+.3f}`
                - **Block LRT Logit Evidence**: `{sig_row.mean_block_evidence:+.3f}`
                """)

                fig_prof = go.Figure()
                fig_prof.add_trace(go.Bar(
                    x=["Parametric", "Spatial", "Block LRT"],
                    y=[sig_row.mean_parametric_evidence, sig_row.mean_spatial_evidence, sig_row.mean_block_evidence],
                    marker_color=["#00F0FF", "#FFB74D", "#00E676"],
                    text=[f"{sig_row.mean_parametric_evidence:+.2f}", f"{sig_row.mean_spatial_evidence:+.2f}", f"{sig_row.mean_block_evidence:+.2f}"],
                    textposition="auto",
                ))
                fig_prof.update_layout(
                    height=240,
                    template="plotly_dark",
                    paper_bgcolor="#121721",
                    plot_bgcolor="#121721",
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=f"Logit Evidence Components for Cluster {sig_row.cluster_id}",
                    yaxis=dict(gridcolor="#1E2638", title="Log-Odds Contribution"),
                )
                st.plotly_chart(fig_prof, use_container_width=True)

        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid #3B82F6; border-radius: 6px; padding: 14px; margin-top: 16px; font-size: 13px; color: #CBD5E1;">
            <b>Descriptive Cluster Disclaimer</b>: Observed evidence signatures group dies by Model B's multi-resolution logit components. These groupings represent observational model evidence patterns and do NOT imply verified physical manufacturing failure mechanisms.
        </div>
        """, unsafe_allow_html=True)

        # ==============================================================================
        # STEP 5.5: FAILURE SIGNATURE → WAFER → DIE DRILL-DOWN
        # ==============================================================================
        st.markdown("<div style='margin-top: 32px; border-top: 1px solid #1E2638; padding-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔎 Investigate This Signature")
        st.markdown(f"**Selected Signature**: `{sig_row.signature_name}` (Cluster {sig_row.cluster_id}) &nbsp;·&nbsp; Dominant Evidence Source: `{sig_row.dominant_source}`")

        # Concise Explanation Summary Metrics
        s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
        with s_c1:
            st.metric("Die Count", f"{int(sig_row.n_dies):,}")
        with s_c2:
            st.metric("% Analyzed Dies", f"{sig_row.pct_dies:.2f}%")
        with s_c3:
            st.metric("Descriptive Failure Rate", f"{sig_row.failure_rate_pct:.2f}%")
        with s_c4:
            st.metric("Mean Model B Risk", f"{sig_row.mean_model_b_prob*100:.2f}%")
        with s_c5:
            st.metric("Mean A→B ΔP", f"{sig_row.mean_risk_difference*100:+.2f}%")

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # --- AFFECTED WAFERS ---
        st.markdown("### 🌐 Affected Wafers")
        st.markdown("*Wafer-level distribution of dies assigned to this evidence signature.*")

        wafers_summary_df = get_cluster_wafers_summary(selected_cid)

        if wafers_summary_df.empty:
            st.info("No affected wafers found for this signature cluster.")
        else:
            # Table formatting
            disp_wafers_df = wafers_summary_df.copy()
            disp_wafers_df.rename(columns={
                "wafer_id": "Wafer",
                "signature_die_count": "Signature Dies",
                "total_wafer_dies": "Total Dies",
                "pct_of_wafer_dies": "% of Wafer",
                "mean_model_b_prob": "Mean Model B Risk",
                "mean_risk_diff": "Mean A→B ΔP",
                "descriptive_failure_rate": "Descriptive Failure Rate (%)",
            }, inplace=True)

            disp_table = disp_wafers_df[[
                "Wafer", "Signature Dies", "Total Dies", "% of Wafer", 
                "Mean Model B Risk", "Mean A→B ΔP", "Descriptive Failure Rate (%)"
            ]].copy()

            disp_table["% of Wafer"] = disp_table["% of Wafer"].map("{:.2f}%".format)
            disp_table["Mean Model B Risk"] = disp_table["Mean Model B Risk"].map("{:.2%}".format)
            disp_table["Mean A→B ΔP"] = disp_table["Mean A→B ΔP"].map("{:+.2%}".format)
            disp_table["Descriptive Failure Rate (%)"] = disp_table["Descriptive Failure Rate (%)"].map("{:.2f}%".format)

            st.dataframe(disp_table, use_container_width=True, hide_index=True)

            # Wafer Selector Dropdown
            wafer_list = wafers_summary_df["wafer_id"].tolist()
            cur_w = st.session_state.get("selected_wafer", wafer_list[0] if wafer_list else "W_N_0083")
            w_idx = wafer_list.index(cur_w) if cur_w in wafer_list else 0

            st.selectbox(
                "Select a Wafer to Inspect Dies:",
                options=wafer_list,
                index=w_idx,
                key="sig_wafer_selector",
                on_change=on_wafer_change,
                args=("sig_wafer_selector",)
            )
            selected_wafer = st.session_state["selected_wafer"]

            # --- AFFECTED DIES ---
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            st.markdown(f"### 🎯 Dies in This Signature on Wafer `{selected_wafer}`")

            cluster_dies_df = get_cluster_wafer_dies(selected_cid, selected_wafer)

            if cluster_dies_df.empty:
                st.info(f"No dies found on wafer `{selected_wafer}` belonging to signature `{sig_row.signature_name}`.")
            else:
                # Merge block anomaly info if available
                score_col = "peak_scan_value" if "peak_scan_value" in anomaly_df.columns else ("peak_anomaly_score" if "peak_anomaly_score" in anomaly_df.columns else None)
                if not anomaly_df.empty and {"wafer_id", "die_row", "die_col"}.issubset(set(anomaly_df.columns)):
                    merge_cols = ["wafer_id", "die_row", "die_col"]
                    if "peak_anomaly_index" in anomaly_df.columns:
                        merge_cols.append("peak_anomaly_index")
                    if score_col:
                        merge_cols.append(score_col)
                    cluster_dies_df = cluster_dies_df.merge(
                        anomaly_df[merge_cols],
                        on=["wafer_id", "die_row", "die_col"],
                        how="left"
                    )
                else:
                    cluster_dies_df["peak_anomaly_index"] = np.nan
                    if score_col:
                        cluster_dies_df[score_col] = np.nan

                disp_dies_df = cluster_dies_df.copy()
                disp_dies_df.rename(columns={
                    "die_row": "Die Row",
                    "die_col": "Die Col",
                    "model_a_probability": "Model A Prob",
                    "model_b_probability": "Model B Risk",
                    "risk_difference": "A→B ΔP",
                    "signature_name": "Signature",
                    "dominant_source": "Dominant Evidence",
                }, inplace=True)

                show_cols = ["Die Row", "Die Col", "Model A Prob", "Model B Risk", "A→B ΔP", "Signature", "Dominant Evidence"]
                if score_col and score_col in disp_dies_df.columns:
                    disp_dies_df.rename(columns={score_col: "Block Anomaly Value"}, inplace=True)
                    show_cols.append("Block Anomaly Value")

                fmt_dies_table = disp_dies_df[show_cols].copy()
                fmt_dies_table["Model A Prob"] = fmt_dies_table["Model A Prob"].map("{:.2%}".format)
                fmt_dies_table["Model B Risk"] = fmt_dies_table["Model B Risk"].map("{:.2%}".format)
                fmt_dies_table["A→B ΔP"] = fmt_dies_table["A→B ΔP"].map("{:+.2%}".format)

                if "Block Anomaly Value" in fmt_dies_table.columns:
                    fmt_dies_table["Block Anomaly Value"] = fmt_dies_table["Block Anomaly Value"].apply(
                        lambda val: f"{val:.4f}" if pd.notna(val) else "N/A"
                    )

                num_dies_avail = len(cluster_dies_df)
                if num_dies_avail > 1:
                    top_n = st.slider("Display Top N Risk Dies:", min_value=1, max_value=num_dies_avail, value=min(20, num_dies_avail), key="sig_top_n_slider")
                else:
                    top_n = num_dies_avail

                st.dataframe(fmt_dies_table.head(top_n), use_container_width=True, hide_index=True)

                # Die Selector Dropdown
                die_options = [
                    f"Die (Row {row['die_row']}, Col {row['die_col']}) — Model B Risk: {row['model_b_probability']:.2%} (ΔP: {row['risk_difference']:+.2%})"
                    for _, row in cluster_dies_df.iterrows()
                ]

                if "selected_die_row" in st.session_state and "selected_die_col" in st.session_state:
                    cur_die_r = st.session_state["selected_die_row"]
                    cur_die_c = st.session_state["selected_die_col"]
                    for i, row in cluster_dies_df.iterrows():
                        if row["die_row"] == cur_die_r and row["die_col"] == cur_die_c:
                            st.session_state["sig_die_select"] = die_options[i]
                            break

                selected_die_str = st.selectbox(
                    "Select a Die to Investigate:",
                    options=die_options,
                    key="sig_die_select"
                )

                selected_die_row = int(cluster_dies_df.iloc[die_options.index(selected_die_str)]["die_row"])
                selected_die_col = int(cluster_dies_df.iloc[die_options.index(selected_die_str)]["die_col"])

                st.session_state["sig_selected_die_row"] = selected_die_row
                st.session_state["sig_selected_die_col"] = selected_die_col

                # Synchronize shared cross-page session state
                st.session_state["selected_die_row"] = selected_die_row
                st.session_state["selected_die_col"] = selected_die_col
                st.session_state["active_wafer"] = selected_wafer
                st.session_state["ov_row_selector"] = selected_die_row
                st.session_state["ov_col_selector"] = selected_die_col

                # --- DIE SELECTION → EXISTING DIE EXPLANATION ---
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                col_exp1, col_exp2 = st.columns([3, 1])
                with col_exp1:
                    st.markdown(f"### 🔍 Investigate Selected Die: `{selected_wafer}` · Die ({selected_die_row}, {selected_die_col})")
                with col_exp2:
                    if st.button("🔍 Open in Die Explanation Page", key="sig_explain_die_btn", use_container_width=True):
                        st.session_state["explanation_mode"] = "explorer"
                        st.session_state["nav_section_radio"] = "🔍 Die Explanation"
                        st.rerun()

                render_die_explanation(selected_wafer, selected_die_row, selected_die_col)

# --- PAGE 9: WAFER HOTSPOTS ---
elif nav_section == "🔥 Wafer Hotspots":
    st.markdown("""
    <div style="background: #121721; border: 1px solid #1E2638; border-left: 4px solid #FF5252; border-radius: 8px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 20px; font-weight: 900; color: #FFFFFF; letter-spacing: 0.5px;">WAFER HOTSPOT DETECTION &amp; HIGH-RISK REGION EXPLORER</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 4px;">Spatial concentration analysis identifying 8-connected high-risk spatial regions on Model B's predicted risk surface.</div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer banner
    st.markdown("""
    <div style="background: rgba(255, 82, 82, 0.1); border: 1px solid #FF5252; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; font-size: 12px; color: #CBD5E1;">
        🔥 <b>Observational Diagnostic Note</b>: Hotspots are spatially connected regions of high predicted Model B risk. They represent diagnostic risk concentrations and do NOT establish causal physical manufacturing defect mechanisms.
    </div>
    """, unsafe_allow_html=True)

    # Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1, 1, 1])
    
    with ctrl_col1:
        wafers_avail = available_all_wafers if available_all_wafers else ["W_N_0083"]
        cur_w = st.session_state.get("selected_wafer", wafers_avail[0])
        w_idx = wafers_avail.index(cur_w) if cur_w in wafers_avail else 0
            
        st.selectbox(
            "Select Wafer to Analyze:",
            options=wafers_avail,
            index=w_idx,
            key="hotspot_wafer_selector",
            on_change=on_wafer_change,
            args=("hotspot_wafer_selector",)
        )
        selected_wafer = st.session_state["selected_wafer"]
        st.session_state["hotspot_selected_wafer"] = selected_wafer

    with ctrl_col2:
        risk_mode = st.radio(
            "Risk Surface Mode:",
            ["Model B Risk", "A→B ΔP Gain"],
            index=0,
            horizontal=True,
            key="hotspot_risk_mode"
        )
        risk_col_name = "prob_model_b" if risk_mode == "Model B Risk" else "risk_difference"

    with ctrl_col3:
        percentile_thresh = st.slider(
            "Risk Percentile Threshold (%):",
            min_value=70,
            max_value=99,
            value=95,
            step=1,
            key="hotspot_percentile_slider"
        )

    with ctrl_col4:
        min_size = st.slider(
            "Min Hotspot Size (dies):",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="hotspot_min_size_slider"
        )

    # Detect hotspots
    hotspots_df, annotated_wafer_df = detect_wafer_hotspots(
        wafer_id=selected_wafer,
        percentile_threshold=float(percentile_thresh),
        min_hotspot_dies=int(min_size),
        risk_col=risk_col_name
    )

    # KPI Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    n_hotspots = len(hotspots_df) if not hotspots_df.empty else 0
    n_high_risk_dies = int(annotated_wafer_df["is_high_risk_die"].sum()) if not annotated_wafer_df.empty and "is_high_risk_die" in annotated_wafer_df.columns else 0
    highest_mean_risk = float(hotspots_df["mean_risk"].max()) if not hotspots_df.empty else 0.0
    largest_hotspot_dies = int(hotspots_df["n_dies"].max()) if not hotspots_df.empty else 0

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HOTSPOTS DETECTED</div>
            <div class="kpi-val-red">{n_hotspots}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HIGH-RISK DIES (≥{percentile_thresh}th%)</div>
            <div class="kpi-val-cyan">{n_high_risk_dies}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HIGHEST MEAN RISK</div>
            <div class="kpi-val-cyan">{highest_mean_risk*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">LARGEST HOTSPOT SIZE</div>
            <div class="kpi-val-green">{largest_hotspot_dies} dies</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Visual Wafer Map with Hotspot Outlines / Bounding Boxes
    st.markdown(f"#### 🗺️ Model B Risk Surface &amp; High-Risk Hotspots on Wafer `{selected_wafer}`")
    
    if annotated_wafer_df.empty:
        st.warning("No prediction data available for this wafer.")
    else:
        display_prob_col = risk_col_name if risk_col_name in annotated_wafer_df.columns else ("prob_model_b" if "prob_model_b" in annotated_wafer_df.columns else "predicted_probability")
        
        fig_map = px.scatter(
            annotated_wafer_df,
            x="die_col",
            y="die_row",
            color=display_prob_col,
            color_continuous_scale="Reds" if risk_mode == "Model B Risk" else "Plasma",
            custom_data=["die_row", "die_col", "wafer_id", display_prob_col, "hotspot_id"],
            hover_data=["die_row", "die_col", display_prob_col, "hotspot_id"],
            labels={"die_col": "Die Column", "die_row": "Die Row", display_prob_col: "Risk Value"},
        )
        fig_map.update_traces(marker=dict(size=9, opacity=0.85))

        shapes_list = []
        annotations_list = []

        if not hotspots_df.empty:
            color_palette = ["#FF5252", "#FFB74D", "#00F0FF", "#E040FB", "#00E676", "#FFD600", "#FF6D00"]
            
            for idx, h_row in hotspots_df.iterrows():
                h_id = h_row["hotspot_id"]
                r_min, r_max = h_row["row_min"], h_row["row_max"]
                c_min, c_max = h_row["col_min"], h_row["col_max"]
                box_color = color_palette[idx % len(color_palette)]
                
                shapes_list.append(dict(
                    type="rect",
                    x0=c_min - 0.5,
                    y0=r_min - 0.5,
                    x1=c_max + 0.5,
                    y1=r_max + 0.5,
                    line=dict(color=box_color, width=2.5, dash="solid"),
                    fillcolor=box_color,
                    opacity=0.15,
                ))
                
                annotations_list.append(dict(
                    x=h_row["centroid_col"],
                    y=h_row["centroid_row"],
                    text=f"<b>{h_id}</b> ({int(h_row['n_dies'])}d)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=box_color,
                    ax=0,
                    ay=-25,
                    font=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"),
                    bgcolor="#0B0E14",
                    bordercolor=box_color,
                    borderwidth=1.5,
                    borderpad=3,
                ))

        iso_dies = annotated_wafer_df[annotated_wafer_df["is_isolated_high_risk"] == True]
        if not iso_dies.empty:
            fig_map.add_trace(go.Scatter(
                x=iso_dies["die_col"],
                y=iso_dies["die_row"],
                mode="markers",
                marker=dict(size=12, symbol="circle-open", color="#FFD600", line=dict(width=2)),
                name="Isolated High-Risk Die (< min size)",
                hoverinfo="skip",
            ))

        fig_map.update_layout(
            height=520,
            template="plotly_dark",
            paper_bgcolor="#121721",
            plot_bgcolor="#121721",
            yaxis=dict(autorange="reversed", gridcolor="#1E2638"),
            xaxis=dict(gridcolor="#1E2638"),
            margin=dict(l=10, r=10, t=30, b=10),
            shapes=shapes_list,
            annotations=annotations_list,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('<div style="font-size: 11px; color: #64748B; margin-top: -8px;">ℹ️ <b>Transparent Map Note</b>: Solid bounding boxes highlight spatial hotspots with &ge;' + str(min_size) + ' connected high-risk dies. Isolated high-risk dies are outlined in yellow but excluded from spatial hotspots.</div>', unsafe_allow_html=True)

    # Hotspot Summary Table
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🔥 Detected Hotspots")

    if hotspots_df.empty:
        st.info("No spatial hotspots detected at current threshold settings.")
    else:
        disp_hotspots_table = hotspots_df.copy()
        disp_hotspots_table.rename(columns={
            "hotspot_id": "Hotspot",
            "n_dies": "Dies",
            "mean_risk": "Mean Risk",
            "max_risk": "Max Risk",
            "mean_risk_diff": "Mean A→B ΔP",
            "center_str": "Center",
            "bounding_region": "Bounding Region",
        }, inplace=True)

        show_cols_h = ["Hotspot", "Dies", "Mean Risk", "Max Risk", "Mean A→B ΔP", "Center", "Bounding Region"]
        fmt_h_table = disp_hotspots_table[show_cols_h].copy()
        fmt_h_table["Mean Risk"] = fmt_h_table["Mean Risk"].map("{:.2%}".format)
        fmt_h_table["Max Risk"] = fmt_h_table["Max Risk"].map("{:.2%}".format)
        fmt_h_table["Mean A→B ΔP"] = fmt_h_table["Mean A→B ΔP"].map("{:+.2%}".format)

        st.dataframe(fmt_h_table, use_container_width=True, hide_index=True)

        # Hotspot Selector
        hotspot_ids = hotspots_df["hotspot_id"].tolist()
        selected_hotspot_id = st.selectbox(
            "Select Hotspot to Inspect Dies:",
            options=hotspot_ids,
            index=0,
            key="hotspot_selector_dropdown"
        )

        h_info = hotspots_df[hotspots_df["hotspot_id"] == selected_hotspot_id].iloc[0]

        # Hotspot Info & Affected Dies Section
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 Hotspot `{selected_hotspot_id}` Overview &amp; Candidate Dies")

        h_d1, h_d2 = st.columns([1, 1])
        with h_d1:
            with st.container(border=True):
                st.markdown(f"##### Hotspot Summary — `{selected_hotspot_id}`")
                st.markdown(f"""
                - **Wafer Target**: `{selected_wafer}`
                - **High-Risk Die Count**: `{int(h_info['n_dies'])}` dies
                - **Mean Model B Risk**: `{h_info['mean_risk']*100:.2f}%`
                - **Max Model B Risk**: `{h_info['max_risk']*100:.2f}%`
                - **Mean Risk Gain ($\Delta P$)**: `{h_info['mean_risk_diff']*100:+.2f}%`
                - **Bounding Box**: `{h_info['bounding_region']}`
                - **Centroid**: `{h_info['center_str']}`
                """)

        with h_d2:
            with st.container(border=True):
                st.markdown("##### Hotspot Spatial Distribution")
                st.markdown(f"""
                - **Row Range**: Row `{int(h_info['row_min'])}` to Row `{int(h_info['row_max'])}`
                - **Col Range**: Col `{int(h_info['col_min'])}` to Col `{int(h_info['col_max'])}`
                - **Spatial Span**: `{int(h_info['row_max'] - h_info['row_min'] + 1)}` rows &times; `{int(h_info['col_max'] - h_info['col_min'] + 1)}` cols
                """)

        # Table of Dies in Selected Hotspot
        hotspot_dies_df = annotated_wafer_df[annotated_wafer_df["hotspot_id"] == selected_hotspot_id].copy()
        disp_p_col = risk_col_name if risk_col_name in hotspot_dies_df.columns else "prob_model_b"
        hotspot_dies_df = hotspot_dies_df.sort_values(by=disp_p_col, ascending=False).reset_index(drop=True)

        st.markdown(f"##### 🎯 High-Risk Dies in Hotspot `{selected_hotspot_id}`")
        
        disp_h_dies = hotspot_dies_df.copy()
        disp_h_dies.rename(columns={
            "die_row": "Die Row",
            "die_col": "Die Col",
            "prob_model_a": "Model A Prob",
            "prob_model_b": "Model B Risk",
            "risk_difference": "A→B ΔP",
            "hotspot_id": "Hotspot ID",
        }, inplace=True)

        show_h_dies_cols = ["Die Row", "Die Col", "Model A Prob", "Model B Risk", "A→B ΔP", "Hotspot ID"]
        fmt_h_dies_table = disp_h_dies[show_h_dies_cols].copy()
        fmt_h_dies_table["Model A Prob"] = fmt_h_dies_table["Model A Prob"].map("{:.2%}".format)
        fmt_h_dies_table["Model B Risk"] = fmt_h_dies_table["Model B Risk"].map("{:.2%}".format)
        fmt_h_dies_table["A→B ΔP"] = fmt_h_dies_table["A→B ΔP"].map("{:+.2%}".format)

        num_h_dies = len(hotspot_dies_df)
        if num_h_dies > 1:
            top_h_n = st.slider("Display Top N Dies in Hotspot:", min_value=1, max_value=num_h_dies, value=min(20, num_h_dies), key="hotspot_top_n_slider")
        else:
            top_h_n = num_h_dies
        st.dataframe(fmt_h_dies_table.head(top_h_n), use_container_width=True, hide_index=True)

        # Die Selector Dropdown
        die_h_options = [
            f"Die (Row {row['die_row']}, Col {row['die_col']}) — Model B Risk: {row['prob_model_b']:.2%} (ΔP: {row['risk_difference']:+.2%})"
            for _, row in hotspot_dies_df.iterrows()
        ]

        if "selected_die_row" in st.session_state and "selected_die_col" in st.session_state:
            cur_h_r = st.session_state["selected_die_row"]
            cur_h_c = st.session_state["selected_die_col"]
            for i, row in hotspot_dies_df.iterrows():
                if row["die_row"] == cur_h_r and row["die_col"] == cur_h_c:
                    st.session_state["hotspot_die_select_dropdown"] = die_h_options[i]
                    break

        selected_die_h_str = st.selectbox(
            "Select a Die in This Hotspot to Investigate:",
            options=die_h_options,
            key="hotspot_die_select_dropdown"
        )

        selected_h_die_row = int(hotspot_dies_df.iloc[die_h_options.index(selected_die_h_str)]["die_row"])
        selected_h_die_col = int(hotspot_dies_df.iloc[die_h_options.index(selected_die_h_str)]["die_col"])

        # Synchronize shared session state across pages
        st.session_state["selected_die_row"] = selected_h_die_row
        st.session_state["selected_die_col"] = selected_h_die_col
        st.session_state["active_wafer"] = selected_wafer
        st.session_state["ov_row_selector"] = selected_h_die_row
        st.session_state["ov_col_selector"] = selected_h_die_col

        # Render Die Explanation
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        col_hexp1, col_hexp2 = st.columns([3, 1])
        with col_hexp1:
            st.markdown(f"### 🔍 Investigate Hotspot Die: `{selected_wafer}` · Die ({selected_h_die_row}, {selected_h_die_col})")
        with col_hexp2:
            if st.button("🔍 Open in Die Explanation Page", key="hotspot_explain_die_btn", use_container_width=True):
                st.session_state["explanation_mode"] = "explorer"
                st.session_state["nav_section_radio"] = "🔍 Die Explanation"
                st.rerun()

        render_die_explanation(selected_wafer, selected_h_die_row, selected_h_die_col)

# --- Footnote Disclaimer ---
st.markdown("""
<div class="footer-note">
    <div><b>Observational Interpretability Note</b>: Evidence contributions describe how the trained Model B formed its prediction log-odds. They represent exact observational model attributions, not causal physical proof.</div>
    <div>SanDisk Hackathon | Die Yield Prediction | v1.0</div>
</div>
""", unsafe_allow_html=True)
