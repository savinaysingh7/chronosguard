"""
ChronosGuard: Interactive Data Drift & Model Decay Detection Dashboard
Full-Stack Streamlit Application with Multi-Tier Drift Analytics,
Label-Free Performance Estimation (CBPE), and Cost-Aware Retraining Console.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generator import generate_all_periods
from src.model.trainer import train_baseline_model, FEATURE_COLUMNS, TARGET_COLUMN
from src.model.evaluator import evaluate_predictions
from src.drift.univariate import calculate_psi, calculate_ks_test, calculate_wasserstein, evaluate_univariate_drift
from src.drift.multivariate import calculate_mmd, evaluate_multivariate_drift
from src.drift.adversarial import evaluate_adversarial_drift
from src.drift.cbpe import estimate_metrics_cbpe
from src.mitigation.cost_model import evaluate_retraining_decision, calculate_operational_cost
from src.mitigation.retrainer import execute_retraining_pipeline

st.set_page_config(
    page_title="ChronosGuard | MLOps Drift & Decay Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, executive appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-stable {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-drift {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing ChronosGuard Core Models & Data...")
def initialize_system():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    batches = generate_all_periods(save_dir=raw_dir, seed=42)
    ref_df = batches["T0_Reference"]
    
    # Proper train/validation split of T0
    np.random.seed(42)
    perm = np.random.permutation(len(ref_df))
    split_idx = int(0.8 * len(ref_df))
    train_df = ref_df.iloc[perm[:split_idx]].copy().reset_index(drop=True)
    val_df   = ref_df.iloc[perm[split_idx:]].copy().reset_index(drop=True)
    
    model, train_meta = train_baseline_model(train_df, FEATURE_COLUMNS, TARGET_COLUMN, random_state=42)
    
    # Out-of-sample baseline evaluation on validation set
    val_X = val_df[FEATURE_COLUMNS].values
    val_y = val_df[TARGET_COLUMN].values
    val_probs = model.predict_proba(val_X)[:, 1]
    val_eval = evaluate_predictions(val_y, val_probs)
    val_cbpe = estimate_metrics_cbpe(val_probs)
    
    # Compute per-sample baseline error rates for fair cross-batch comparison
    n_val_neg = val_eval["tn"] + val_eval["fp"]
    n_val_pos = val_eval["tp"] + val_eval["fn"]
    baseline_fpr = val_eval["fp"] / n_val_neg if n_val_neg > 0 else 0.0
    baseline_fnr = val_eval["fn"] / n_val_pos if n_val_pos > 0 else 0.0
    
    return batches, model, val_eval, val_cbpe, baseline_fpr, baseline_fnr

batches, baseline_model, ref_eval, ref_cbpe, baseline_fpr, baseline_fnr = initialize_system()
ref_df = batches["T0_Reference"]
ref_X = ref_df[FEATURE_COLUMNS].values

# Sidebar Navigation & Stream Selection
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("ChronosGuard Engine")
st.sidebar.caption("Continuous Model Reliability & Drift Mitigation")

batch_options = {
    "T1_Normal": "T1: In-Distribution Production Stream (Nominal)",
    "T2_Gradual_Drift": "T2: Gradual Covariate Drift (Inflation Shock)",
    "T3_Sudden_Shock": "T3: Sudden Covariate Shock (Recession Crisis)",
    "T4_Concept_Drift": "T4: Concept Drift (Behavioral Shift P(Y|X))"
}

selected_batch_key = st.sidebar.selectbox(
    "Active Production Batch",
    options=list(batch_options.keys()),
    format_func=lambda x: batch_options[x]
)

current_df = batches[selected_batch_key]
curr_X = current_df[FEATURE_COLUMNS].values
curr_y = current_df[TARGET_COLUMN].values
curr_probs = baseline_model.predict_proba(curr_X)[:, 1]

# Financial Parameters Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Cost Model Configuration")
cost_fn = st.sidebar.slider("Cost of False Negative (Default)", 2000, 15000, 6500, step=500, format="$%d")
cost_fp = st.sidebar.slider("Cost of False Positive (Friction)", 50, 1000, 250, step=50, format="$%d")
retrain_cost = st.sidebar.slider("Retraining Pipeline Expense", 500, 5000, 1500, step=250, format="$%d")

# Header Section
st.markdown('<div class="main-title">ChronosGuard: Data Drift & Model Decay Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Active Monitoring View: <b>{batch_options[selected_batch_key]}</b> | Reference Baseline: <b>T0_Reference (6,000 samples)</b></div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "📈 Univariate Drift (PSI / KS)",
    "🌐 Multivariate Drift (MMD / Domain Classifier)",
    "🎯 Label-Free Performance (CBPE)",
    "💰 Cost-Aware Mitigation & Retrain",
    "🧪 Live Drift Simulation Studio"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    # Run key evaluations for active batch
    uni_summary = evaluate_univariate_drift(ref_df, current_df, FEATURE_COLUMNS)
    max_psi = float(uni_summary["psi"].max())
    n_drifted = int((uni_summary["status"] == "DRIFT_DETECTED").sum())
    
    cbpe_curr = estimate_metrics_cbpe(curr_probs)
    gt_curr = evaluate_predictions(curr_y, curr_probs)
    
    cost_decision = evaluate_retraining_decision(
        current_fp=gt_curr["fp"],
        current_fn=gt_curr["fn"],
        baseline_fp=baseline_fpr * (gt_curr["tn"] + gt_curr["fp"]),
        baseline_fn=baseline_fnr * (gt_curr["tp"] + gt_curr["fn"]),
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        retrain_expense=retrain_cost
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if max_psi < 0.1 and gt_curr["roc_auc"] >= 0.85:
            st.metric("System Health", "STABLE", delta="Nominal", delta_color="normal")
        elif max_psi < 0.2:
            st.metric("System Health", "WARNING", delta="Moderate Shift", delta_color="off")
        else:
            st.metric("System Health", "DECAY DETECTED", delta="-Critical Drift", delta_color="inverse")
            
    with col2:
        auc_drop = gt_curr["roc_auc"] - ref_eval["roc_auc"]
        st.metric("Realized ROC-AUC", f"{gt_curr['roc_auc']:.3f}", delta=f"{auc_drop:+.3f}")
        
    with col3:
        cbpe_drop = cbpe_curr["estimated_auc"] - ref_cbpe["estimated_auc"]
        st.metric("CBPE Estimated AUC", f"{cbpe_curr['estimated_auc']:.3f}", delta=f"{cbpe_drop:+.3f} (Label-Free)")
        
    with col4:
        st.metric("Drifted Features", f"{n_drifted} / {len(FEATURE_COLUMNS)}", delta=f"Max PSI: {max_psi:.3f}")
        
    with col5:
        st.metric("Decay Economic Loss", f"${cost_decision['decay_loss']:,.0f}", delta=f"{cost_decision['recommendation']}")

    st.markdown("---")
    
    # Historical / Multi-Period Trajectory View
    st.subheader("Multi-Period Trajectory (T0 Reference through T4 Concept Drift)")
    
    batch_names_ordered = ["T0_Reference", "T1_Normal", "T2_Gradual_Drift", "T3_Sudden_Shock", "T4_Concept_Drift"]
    period_labels = {
        "T0_Reference": "Q1 2023 (Baseline)",
        "T1_Normal": "Q2 2023 (Normal)",
        "T2_Gradual_Drift": "Q3 2023 (Inflation)",
        "T3_Sudden_Shock": "Q4 2023 (Shock)",
        "T4_Concept_Drift": "Q1 2024 (Concept Shift)"
    }
    traj_records = []
    
    for b_name in batch_names_ordered:
        b_df = batches[b_name]
        b_X = b_df[FEATURE_COLUMNS].values
        b_y = b_df[TARGET_COLUMN].values
        b_p = baseline_model.predict_proba(b_X)[:, 1]
        
        b_gt = evaluate_predictions(b_y, b_p)
        b_cbpe = estimate_metrics_cbpe(b_p)
        b_uni = evaluate_univariate_drift(ref_df, b_df, FEATURE_COLUMNS)
        
        traj_records.append({
            "Period": period_labels.get(b_name, b_name),
            "Realized AUC": b_gt["roc_auc"],
            "CBPE Estimated AUC": b_cbpe["estimated_auc"],
            "Realized F1": b_gt["f1"],
            "CBPE Estimated F1": b_cbpe["estimated_f1"],
            "Max PSI": float(b_uni["psi"].max()),
            "Default Rate": b_gt["default_rate"]
        })
        
    traj_df = pd.DataFrame(traj_records)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_perf = px.line(
            traj_df, x="Period", y=["Realized AUC", "CBPE Estimated AUC"],
            markers=True,
            title="Model Performance Decay: Realized vs Label-Free CBPE Estimate",
            labels={"value": "ROC-AUC Score", "variable": "Metric Type"},
            color_discrete_map={"Realized AUC": "#EF4444", "CBPE Estimated AUC": "#3B82F6"}
        )
        fig_perf.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_perf, use_container_width=True)
        
    with col_t2:
        fig_psi = px.bar(
            traj_df, x="Period", y="Max PSI",
            title="Maximum Population Stability Index (PSI) Across Periods",
            labels={"Max PSI": "Max Feature PSI"},
            color="Max PSI",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"]
        )
        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warn (0.1)")
        fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Drift (0.2)")
        st.plotly_chart(fig_psi, use_container_width=True)

# ==========================================
# TAB 2: UNIVARIATE DRIFT (PSI / KS / WASSERSTEIN)
# ==========================================
with tab2:
    st.subheader(f"Univariate Distribution Shift Analysis ({selected_batch_key})")
    
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        selected_feature = st.selectbox("Select Feature to Inspect", options=FEATURE_COLUMNS)
        
        feat_psi, psi_breakdown = calculate_psi(
            ref_df[selected_feature].to_numpy(dtype=float),
            current_df[selected_feature].to_numpy(dtype=float)
        )
        feat_ks_stat, feat_ks_p = calculate_ks_test(
            ref_df[selected_feature].to_numpy(dtype=float),
            current_df[selected_feature].to_numpy(dtype=float)
        )
        feat_wass = calculate_wasserstein(
            ref_df[selected_feature].to_numpy(dtype=float),
            current_df[selected_feature].to_numpy(dtype=float)
        )
        
        st.markdown(f"**Feature Metrics:** `{selected_feature}`")
        st.metric("Population Stability Index (PSI)", f"{feat_psi:.4f}")
        st.metric("KS-Test Statistic (D)", f"{feat_ks_stat:.4f}", delta=f"p-val: {feat_ks_p:.2e}")
        st.metric("Normalized Wasserstein Distance", f"{feat_wass:.4f}")
        
    with col_u2:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ref_df[selected_feature],
            name="Reference (T0)",
            histnorm="probability density",
            opacity=0.55,
            marker_color="#3B82F6"
        ))
        fig_dist.add_trace(go.Histogram(
            x=current_df[selected_feature],
            name=f"Current ({selected_batch_key})",
            histnorm="probability density",
            opacity=0.55,
            marker_color="#EF4444"
        ))
        fig_dist.update_layout(
            barmode="overlay",
            title=f"Distribution Comparison: {selected_feature} (Ref vs Current)",
            xaxis_title=selected_feature,
            yaxis_title="Probability Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    st.markdown("### Full Feature Drift Scorecard")
    st.dataframe(
        uni_summary[["feature", "psi", "ks_statistic", "ks_p_value", "wasserstein_dist_norm", "status"]],
        use_container_width=True
    )

# ==========================================
# TAB 3: MULTIVARIATE DRIFT (MMD & ADVERSARIAL)
# ==========================================
with tab3:
    st.subheader(f"Multivariate Drift Engine ({selected_batch_key})")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 1. Maximum Mean Discrepancy (MMD - RBF Kernel)")
        st.markdown("""
        MMD maps multivariate distributions into a Reproducing Kernel Hilbert Space (RKHS) 
        to detect complex non-linear correlations and cross-feature covariances that univariate tests miss.
        """)
        with st.spinner("Computing MMD & Permutation Test..."):
            multi_res = evaluate_multivariate_drift(ref_X, curr_X, n_permutations=40)
            
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("MMD Score", f"{multi_res['mmd_score']:.5f}")
        m_col2.metric("Permutation p-value", f"{multi_res['p_value']:.3f}", delta=multi_res['status'])
        
    with col_m2:
        st.markdown("#### 2. Adversarial Domain Discriminator")
        st.markdown("""
        Trains a domain classifier to distinguish between Reference ($y=0$) and Streaming Data ($y=1$).
        $AUC \\approx 0.50$ implies identical distributions; $AUC > 0.65$ proves multivariate drift.
        """)
        with st.spinner("Training Adversarial Classifier..."):
            adv_res = evaluate_adversarial_drift(ref_df, current_df, FEATURE_COLUMNS)
            
        a_col1, a_col2 = st.columns(2)
        a_col1.metric("Discriminator AUC", f"{adv_res['adversarial_auc']:.4f}")
        a_col2.metric("Status", adv_res['status'])
        
    st.markdown("#### Primary Drivers of Multivariate Drift")
    adv_feat_df = pd.DataFrame(adv_res["feature_importances"])
    fig_adv = px.bar(
        adv_feat_df, x="importance", y="feature", orientation="h",
        title="Feature Contribution to Domain Separability (Top Drift Drivers)",
        labels={"importance": "Discriminator Importance Score", "feature": "Feature Name"},
        color="importance",
        color_continuous_scale="Viridis"
    )
    fig_adv.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_adv, use_container_width=True)

# ==========================================
# TAB 4: LABEL-FREE PERFORMANCE ESTIMATION (CBPE)
# ==========================================
with tab4:
    st.subheader("Novelty: Label-Free Performance Monitoring via CBPE")
    st.info("""
    **The Real-World Dilemma:** In financial credit and loan risk, true default labels take 90-180 days to observe. 
    Waiting for ground-truth labels leaves institutions vulnerable to silent decay.
    **The Solution:** ChronosGuard utilizes **Confidence-Based Performance Estimation (CBPE)** to compute 
    expected confusion matrices and metrics directly from calibrated probability fields before a single label arrives.
    """)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("CBPE Estimated Accuracy", f"{cbpe_curr['estimated_accuracy']:.3f}", 
                  delta=f"Realized: {gt_curr['accuracy']:.3f}")
    col_c2.metric("CBPE Estimated F1-Score", f"{cbpe_curr['estimated_f1']:.3f}", 
                  delta=f"Realized: {gt_curr['f1']:.3f}")
    col_c3.metric("CBPE Estimated ROC-AUC", f"{cbpe_curr['estimated_auc']:.3f}", 
                  delta=f"Realized: {gt_curr['roc_auc']:.3f}")
                  
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("##### Label-Free Expected Confusion Matrix (CBPE)")
        cbpe_cm_df = pd.DataFrame({
            "Predicted Default (1)": [cbpe_curr["expected_tp"], cbpe_curr["expected_fp"]],
            "Predicted Non-Default (0)": [cbpe_curr["expected_fn"], cbpe_curr["expected_tn"]]
        }, index=["Actual Positive (E)", "Actual Negative (E)"])
        st.table(cbpe_cm_df.style.format("{:,.1f}"))
        
    with col_cm2:
        st.markdown("##### Realized Confusion Matrix (Ground Truth)")
        gt_cm_df = pd.DataFrame({
            "Predicted Default (1)": [gt_curr["tp"], gt_curr["fp"]],
            "Predicted Non-Default (0)": [gt_curr["fn"], gt_curr["tn"]]
        }, index=["Actual Positive", "Actual Negative"])
        st.table(gt_cm_df.style.format("{:,d}"))

# ==========================================
# TAB 5: COST-AWARE MITIGATION & RETRAINING CONSOLE
# ==========================================
with tab5:
    st.subheader("Novelty: Cost-Aware Mitigation & Automated Retraining Engine")
    st.markdown("""
    Instead of naively retraining whenever statistical drift exceeds an arbitrary threshold, 
    ChronosGuard quantifies the **Economic Loss of Decay** against **Retraining Compute & Governance Overhead**.
    """)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Batch Operational Loss", f"${cost_decision['current_loss']:,.2f}")
    col_r2.metric("Decay Loss Attributable to Drift", f"${cost_decision['decay_loss']:,.2f}")
    col_r3.metric("Net Expected Retrain ROI", f"${cost_decision['net_expected_roi']:,.2f}", 
                  delta=f"{cost_decision['roi_ratio']:.1f}x Payback")
                  
    if cost_decision["should_retrain"]:
        st.error(f"🚨 **Recommendation: {cost_decision['recommendation']}** — {cost_decision['reason']}")
    else:
        st.success(f"✅ **Recommendation: {cost_decision['recommendation']}** — {cost_decision['reason']}")
        
    st.markdown("---")
    st.subheader("Automated Retraining Simulator")
    st.caption("Trigger an adaptive sliding-window retraining pipeline blending reference baseline with current streaming data.")
    
    if st.button("🚀 Trigger Automated Retraining Pipeline", type="primary"):
        with st.spinner("Executing sliding-window retraining & re-calibration..."):
            retrain_res = execute_retraining_pipeline(
                reference_df=ref_df,
                drifted_df=current_df,
                feature_cols=FEATURE_COLUMNS,
                target_col=TARGET_COLUMN,
                cost_fp=cost_fp,
                cost_fn=cost_fn
            )
            
        st.success("Retraining Pipeline Successfully Completed!")
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        f1_gain = retrain_res["post_f1"] - gt_curr["f1"]
        auc_gain = retrain_res["post_roc_auc"] - gt_curr["roc_auc"]
        cost_recovered = cost_decision["current_loss"] - retrain_res["post_operational_cost"]
        
        rc1.metric("Post-Retrain ROC-AUC", f"{retrain_res['post_roc_auc']:.3f}", delta=f"{auc_gain:+.3f} Recovery")
        rc2.metric("Post-Retrain F1-Score", f"{retrain_res['post_f1']:.3f}", delta=f"{f1_gain:+.3f} Recovery")
        rc3.metric("Post-Retrain Brier Loss", f"{retrain_res['post_brier']:.4f}")
        rc4.metric("Financial Loss Reduction", f"${cost_recovered:,.0f} Saved")

# ==========================================
# TAB 6: LIVE DRIFT SIMULATION STUDIO
# ==========================================
with tab6:
    st.subheader("🧪 Live Drift Injection & Simulation Sandbox")
    st.markdown("Interactively synthesize custom economic shocks and test the detector sensitivity in real time.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_cov = st.slider("Covariate Shift (Inflation & DTI)", 0.0, 2.0, 0.5, step=0.1)
    with sim_col2:
        sim_shock = st.slider("Macro Shock (Unemployment/Liquidity)", 0.0, 2.0, 0.0, step=0.1)
    with sim_col3:
        sim_conc = st.slider("Concept Drift (Structural Default Shift)", 0.0, 2.0, 0.0, step=0.1)
        
    if st.button("Generate Synthetic Stream Batch"):
        with st.spinner("Generating custom stream & running real-time analytics..."):
            sim_df = generate_credit_batch(
                n_samples=2500,
                batch_name="Custom_Simulated_Stream",
                period_id=99,
                covariate_drift_factor=sim_cov,
                shock_factor=sim_shock,
                concept_drift_factor=sim_conc
            )
            sim_X = sim_df[FEATURE_COLUMNS].values
            sim_y = sim_df[TARGET_COLUMN].values
            sim_p = baseline_model.predict_proba(sim_X)[:, 1]
            
            sim_uni = evaluate_univariate_drift(ref_df, sim_df, FEATURE_COLUMNS)
            sim_cbpe = estimate_metrics_cbpe(sim_p)
            sim_gt = evaluate_predictions(sim_y, sim_p)
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Simulated Batch Default Rate", f"{sim_df['is_default'].mean():.2%}")
            sc2.metric("Max Feature PSI", f"{sim_uni['psi'].max():.4f}")
            sc3.metric("Simulated Realized AUC", f"{sim_gt['roc_auc']:.3f}", delta=f"CBPE: {sim_cbpe['estimated_auc']:.3f}")
            
            st.dataframe(sim_uni[["feature", "psi", "ks_statistic", "status"]], use_container_width=True)
