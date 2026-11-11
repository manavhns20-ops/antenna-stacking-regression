"""
Streamlit Web Application: ANTENNA AI — STACKING REGRESSION SYSTEM
Course: EC-101 | Academic Project: Stacking Regression for Antenna Design
Student: Manav Hans (Roll No: 26BCS063), NIT Hamirpur
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project root path setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import AntennaPreprocessor
from src.stacking import MultiOutputStackingRegressor
from src.predict import AntennaPredictor


# Page configuration
st.set_page_config(
    page_title="ANTENNA AI — STACKING REGRESSION SYSTEM",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean engineering dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 18px;
        border-left: 5px solid #17b978;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .student-badge {
        background-color: #e8f4f8;
        border: 1px solid #c4e3f3;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    model_path = os.path.join(PROJECT_ROOT, "models/saved_model/final_stacking_model.joblib")
    preproc_path = os.path.join(PROJECT_ROOT, "models/saved_model/preprocessor.joblib")
    return AntennaPredictor(model_path=model_path, preprocessor_path=preproc_path)


@st.cache_data
def load_metrics():
    test_metrics_path = os.path.join(PROJECT_ROOT, "results/metrics/test_summary_metrics.csv")
    meta_study_path = os.path.join(PROJECT_ROOT, "results/metrics/meta_model_study_comparison.csv")
    per_target_path = os.path.join(PROJECT_ROOT, "results/metrics/test_per_target_metrics.csv")

    summary_df = pd.read_csv(test_metrics_path) if os.path.exists(test_metrics_path) else None
    meta_df = pd.read_csv(meta_study_path) if os.path.exists(meta_study_path) else None
    per_target_df = pd.read_csv(per_target_path) if os.path.exists(per_target_path) else None
    return summary_df, meta_df, per_target_df


def main():
    predictor = load_predictor()
    summary_df, meta_df, per_target_df = load_metrics()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/satellite-antenna.png", width=70)
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Module:",
        ["Dashboard", "Antenna Performance Predictor", "Model Analysis", "About Project"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Academic Information:**
    - **Student:** Manav Hans
    - **Roll No:** 26BCS063
    - **Institute:** NIT Hamirpur
    - **Course:** EC-101
    - **Category:** Multi-Classical Regression
    - **Algorithm:** Stacking Regression
    """)

    # -------------------------------------------------------------
    # SECTION 1: DASHBOARD
    # -------------------------------------------------------------
    if section == "Dashboard":
        st.markdown("<p class='main-header'>📡 ANTENNA AI — STACKING REGRESSION SYSTEM</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Forward Regression Engine: 10 Antenna Design Variables → 35 Performance Indicators</p>", unsafe_allow_html=True)

        st.markdown("""
        <div class='student-badge'>
            <b>Academic Project Allotment (EC-101):</b> Manav Hans (Roll No: <b>26BCS063</b>) | Department of Electronics & Communication Engineering, NIT Hamirpur | <b>Assigned Algorithm:</b> Stacking Regression with Meta-Model Study
        </div>
        """, unsafe_allow_html=True)

        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Input Variables", value="10 (DV1–DV10)")
        with col2:
            st.metric(label="Target Outputs", value="35 (P1_F1–P7_F5)")
        with col3:
            stack_mae = summary_df.loc[summary_df["Model"].str.contains("Stacking"), "MAE"].values[0] if summary_df is not None else 0.9443
            st.metric(label="Stacking Test MAE", value=f"{stack_mae:.4f}", delta="-0.109 vs RF", delta_color="inverse")
        with col4:
            stack_rmse = summary_df.loc[summary_df["Model"].str.contains("Stacking"), "RMSE"].values[0] if summary_df is not None else 1.6059
            st.metric(label="Stacking Test RMSE", value=f"{stack_rmse:.4f}")

        st.markdown("### 🏆 Ensemble Performance Highlights")
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.write("**Model Benchmark Comparison Table (Untouched Test Partition):**")
            if summary_df is not None:
                st.dataframe(
                    summary_df[["Model", "MAE", "RMSE", "R2", "Training_Time_s", "Prediction_Time_s"]].style.highlight_min(
                        subset=["MAE", "RMSE"], color="#c8e6c9"
                    ).highlight_max(subset=["R2"], color="#c8e6c9"),
                    use_container_width=True
                )
            st.caption("✅ The Stacking Regressor achieves lowest Mean Absolute Error across all models.")

        with col_right:
            plot_path = os.path.join(PROJECT_ROOT, "results/plots/mae_comparison.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Figure 1: Mean Absolute Error (MAE) Comparison across Models")

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            p2 = os.path.join(PROJECT_ROOT, "results/plots/base_vs_stacking.png")
            if os.path.exists(p2):
                st.image(p2, caption="Figure 2: Base Models vs Stacking Multi-Metric Comparison")
        with col_g2:
            p3 = os.path.join(PROJECT_ROOT, "results/plots/meta_model_comparison.png")
            if os.path.exists(p3):
                st.image(p3, caption="Figure 3: Meta-Model Study: Ridge vs OLS vs ElasticNet")

    # -------------------------------------------------------------
    # SECTION 2: ANTENNA PERFORMANCE PREDICTOR
    # -------------------------------------------------------------
    elif section == "Antenna Performance Predictor":
        st.markdown("<p class='main-header'>⚙️ Antenna Performance Predictor</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Enter 10 antenna geometric & physical design variables to compute all 35 performance outputs.</p>", unsafe_allow_html=True)

        st.markdown("#### Step 1: Configure Antenna Design Variables (DV1–DV10)")
        
        # Load dataset to get exact min, max, mean for realistic sliders
        dataset_path = os.path.join(PROJECT_ROOT, "data/dataset.csv")
        df_raw = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else None

        c1, c2, c3 = st.columns(3)
        input_vals = {}

        dv_info = [
            ("DV1", "Patch Length / Substrate Dimension (mm)", 35.0, 85.0, 58.12),
            ("DV2", "Substrate Height / Dielectric Thickness (mm)", 0.40, 0.90, 0.63),
            ("DV3", "Substrate Relative Permittivity (εr)", 45.0, 75.0, 62.30),
            ("DV4", "Feed Line Width / Impedance Match (mm)", 1.20, 3.20, 2.32),
            ("DV5", "Slot / Notch Length (mm)", 8.0, 28.0, 19.04),
            ("DV6", "Ground Plane Width (mm)", 40.0, 110.0, 70.11),
            ("DV7", "Ground Plane Length (mm)", 38.0, 88.0, 60.74),
            ("DV8", "Resonator Spacing / Gap (mm)", 26.0, 82.0, 47.87),
            ("DV9", "Feed Offset / Position (mm)", 30.0, 80.0, 52.10),
            ("DV10", "Total Radiating Volume / Footprint (mm³)", 1500.0, 9300.0, 4435.0),
        ]

        for i, (k, label, min_v, max_v, def_v) in enumerate(dv_info):
            target_col = c1 if i < 3 else (c2 if i < 6 else c3)
            with target_col:
                input_vals[k] = st.number_input(
                    label=f"{k}: {label}",
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(def_v),
                    step=0.01 if max_v < 10 else 1.0,
                    format="%.4f" if max_v < 10 else "%.2f"
                )

        st.markdown("---")
        btn_col, _ = st.columns([1, 4])
        with btn_col:
            predict_clicked = st.button("🚀 Predict Performance", use_container_width=True)

        if predict_clicked:
            with st.spinner("Generating 35 performance outputs via Stacking Regression Engine..."):
                preds_35_df = predictor.predict(input_vals)
                freq_table = predictor.predict_formatted_by_frequency(input_vals)

            st.success("✅ Prediction Successfully Generated!")

            # Highlight key antenna performance parameters
            st.markdown("#### Key Resonance Highlights:")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                s11_f1 = preds_35_df["P1_F1"].values[0]
                st.metric("Return Loss S11 (F1)", f"{s11_f1:.2f} dB", "Good Matching" if s11_f1 < -10 else "High Reflection")
            with k2:
                bw_f1 = preds_35_df["P2_F1"].values[0]
                st.metric("Bandwidth Factor P2 (F1)", f"{bw_f1:.4f}")
            with k3:
                gain_f1 = preds_35_df["P4_F1"].values[0]
                st.metric("Gain P4 (F1)", f"{gain_f1:.2f} dBi")
            with k4:
                eff_f1 = preds_35_df["P5_F1"].values[0]
                st.metric("Radiation Efficiency (F1)", f"{eff_f1:.1f} %")

            # Structured Output Matrix
            st.markdown("#### Complete 35 Performance Outputs (Grouped by Frequencies F1–F5):")
            st.markdown("""
            * **P1**: Return Loss / Reflection Coefficient $S_{11}$ (dB)
            * **P2**: VSWR / Fractional Bandwidth Magnitude
            * **P3**: Directivity ($D$) (dBi)
            * **P4**: Antenna Realized Gain ($G$) (dBi)
            * **P5**: Radiation Efficiency ($\eta_{rad}$) (%)
            * **P6**: Half-Power Beamwidth (HPBW E-plane) (deg)
            * **P7**: Half-Power Beamwidth (HPBW H-plane) (deg)
            """)

            st.dataframe(
                freq_table.style.format({
                    "P1": "{:.2f} dB",
                    "P2": "{:.4f}",
                    "P3": "{:.2f} dBi",
                    "P4": "{:.2f} dBi",
                    "P5": "{:.2f} %",
                    "P6": "{:.2f}°",
                    "P7": "{:.2f}°",
                }),
                use_container_width=True
            )

            # CSV Download Button
            csv_data = preds_35_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_data,
                file_name="antenna_predicted_performance_35.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------
    # SECTION 3: MODEL ANALYSIS
    # -------------------------------------------------------------
    elif section == "Model Analysis":
        st.markdown("<p class='main-header'>📊 Model & Meta-Model Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>In-depth empirical investigation of base regressors, out-of-fold stacking, and meta-model regularization.</p>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Meta-Model Study", "Scatter & Residual Analysis", "Feature Contributions"])

        with tab1:
            st.subheader("Section 7 Meta-Model Study: Regularization Comparison")
            st.markdown("""
            Stacking regression utilizes a level-1 meta-model to learn the optimal fusion of level-0 predictions.
            Here we compare **Ridge Regression (L2 Regularized)**, **Linear Regression (Ordinary Least Squares)**,
            and **ElasticNet** on identical Out-of-Fold (OOF) feature matrices.
            """)

            if meta_df is not None:
                st.dataframe(meta_df, use_container_width=True)

            p_meta = os.path.join(PROJECT_ROOT, "results/plots/meta_model_comparison.png")
            if os.path.exists(p_meta):
                st.image(p_meta, caption="Figure 4: Validation and Test Error across Meta-Models")

            st.info("""
            **Key Academic Finding:**
            Standard Ordinary Least Squares (Linear Regression) suffers from collinearity among base models (RF, ET, and GBR outputs are highly correlated), leading to variance inflation.
            **Ridge Regression with $\\alpha = 20.0$** introduces an L2 penalty, stabilizing the weights and yielding the lowest overall Mean Absolute Error (MAE = 0.9443).
            """)

        with tab2:
            st.subheader("Goodness of Fit and Residual Dynamics")
            c1, c2 = st.columns(2)
            with c1:
                p_act = os.path.join(PROJECT_ROOT, "results/plots/actual_vs_predicted.png")
                if os.path.exists(p_act):
                    st.image(p_act, caption="Figure 5: Actual vs Predicted Scatter Plots (Test Set)")
            with c2:
                p_res = os.path.join(PROJECT_ROOT, "results/plots/residual_distribution.png")
                if os.path.exists(p_res):
                    st.image(p_res, caption="Figure 6: Residual Distribution (Mean ~ 0, Normal Kurtosis)")

            p_r2 = os.path.join(PROJECT_ROOT, "results/plots/individual_r2_scores.png")
            if os.path.exists(p_r2):
                st.image(p_r2, caption="Figure 7: R2 Score per Antenna Performance Indicator")

        with tab3:
            st.subheader("Feature Importance & Correlation Architecture")
            c3, c4 = st.columns(2)
            with c3:
                p_fi = os.path.join(PROJECT_ROOT, "results/plots/feature_importance.png")
                if os.path.exists(p_fi):
                    st.image(p_fi, caption="Figure 8: Feature Importance of Design Variables (DV1-DV10)")
            with c4:
                p_corr = os.path.join(PROJECT_ROOT, "results/plots/correlation_heatmap.png")
                if os.path.exists(p_corr):
                    st.image(p_corr, caption="Figure 9: Correlation Matrix (DV1-DV10 vs Outputs)")

    # -------------------------------------------------------------
    # SECTION 4: ABOUT PROJECT
    # -------------------------------------------------------------
    elif section == "About Project":
        st.markdown("<p class='main-header'>📖 Academic Project Documentation</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>EC-101: AI, Inverse AI Regression and Optimization for Antenna Design</p>", unsafe_allow_html=True)

        st.markdown("""
        ### 1. Student & Project Identification
        - **Student Name:** Manav Hans
        - **Roll Number:** 26BCS063
        - **Institution:** National Institute of Technology (NIT) Hamirpur
        - **Department:** Electronics and Communication Engineering
        - **Course:** EC-101 [B.Tech Semester 1]
        - **Assigned Topic:** **Stacking Regression for Antenna Design**
        - **Category:** Multi-Classical Regression (Item Allotment #63)

        ---

        ### 2. Physical Antenna Problem Definition
        Modern multiband microstrip antennas are governed by complex electromagnetic field equations.
        Full-wave numerical solvers (e.g., CST Microwave Studio, HFSS, Ansys) require extensive computational time
        (often hours per simulation).
        This project builds a **Machine Learning Forward Surrogate Model** that instantaneously predicts **35 antenna performance metrics** from **10 design parameters**.

        * **Inputs (10 Design Variables):**
          - `DV1, DV2, DV3`: Patch dimensions and substrate dielectric permittivity governing resonance frequencies $F_1 \dots F_5$ and Directivity/Gain ($P_3, P_4$).
          - `DV4`: Feedline geometry controlling input impedance, return loss ($P_1$), and bandwidth ($P_2$).
          - `DV5 – DV10`: Slot length, ground plane dimensions, and radiator volume governing efficiency ($P_5$) and radiation beamwidths ($P_6, P_7$).

        * **Outputs (35 Performance Indicators):**
          - 7 key metrics evaluated at 5 discrete operating frequencies: $P1\_F1 \dots P7\_F5$.

        ---

        ### 3. Stacking Regression Architecture
        Unlike simple voting or blending, **Stacking Regression (Wolpert, 1992)** trains a meta-model on **out-of-fold (OOF)** predictions:
        1. **Level-0 Base Learners:**
           - Random Forest Regressor (non-linear bagging)
           - Extra Trees Regressor (extremely randomized ensembles)
           - Gradient Boosting Regressor (sequential boosting)
           - K-Nearest Neighbors Regressor (local non-parametric distance)
           - Support Vector Regressor (RBF kernel maximum margin)
           - Ridge Regressor (linear regularized baseline)
        2. **K-Fold Cross-Validation:**
           - The 70% training partition is divided into 5 folds.
           - Base models generate unbiased OOF predictions for held-out folds.
        3. **Level-1 Meta-Learner:**
           - Ridge Regression learns optimal regularization coefficients to blend base model outputs per target without overfitting.
        """)


if __name__ == "__main__":
    main()
