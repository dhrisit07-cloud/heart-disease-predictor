import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────
# EXACT COLUMN ORDER — must match training
# ─────────────────────────────────────────
# FIXED — includes ChestPainType_TA
FEATURE_COLUMNS = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]

# ─────────────────────────────────────────
# LOAD MODEL + SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('best_xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.caption(
    "Built with XGBoost · Trained on UCI Heart Disease Dataset (918 patients) · "
    "For research purposes only — not a clinical diagnostic tool"
)
st.divider()

if not model_loaded:
    st.error(
        "Model files not found. Make sure `best_xgb_model.joblib` and "
        "`scaler.joblib` are in the same folder as this app.py file."
    )
    st.stop()

# ─────────────────────────────────────────
# SIDEBAR — patient input
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Patient Details")
    st.caption("Enter the patient's clinical information below.")

    st.subheader("Clinical Measurements")

    age = st.slider("Age", min_value=20, max_value=100, value=54, step=1)

    sex = st.radio("Sex", options=["Male", "Female"], horizontal=True)

    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=["ATA — Atypical Angina",
                 "NAP — Non-Anginal Pain",
                 "ASY — Asymptomatic",
                 "TA  — Typical Angina"],
        index=2
    )
    chest_pain_code = chest_pain.split(" — ")[0].strip()

    resting_bp = st.slider(
        "Resting Blood Pressure (mmHg)",
        min_value=80, max_value=200, value=130, step=1
    )

    cholesterol = st.slider(
        "Cholesterol (mg/dL)",
        min_value=100, max_value=600, value=237, step=1
    )
    if cholesterol == 0:
        st.warning("Cholesterol 0 is not valid — using median (237)")
        cholesterol = 237

    fasting_bs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL?",
        options=["No (0)", "Yes (1)"],
        horizontal=True
    )
    fasting_bs_code = 1 if "Yes" in fasting_bs else 0

    resting_ecg = st.selectbox(
        "Resting ECG",
        options=["Normal", "ST — ST-T wave abnormality", "LVH — Left ventricular hypertrophy"],
        index=0
    )
    resting_ecg_code = resting_ecg.split(" — ")[0].strip()

    max_hr = st.slider(
        "Max Heart Rate Achieved",
        min_value=60, max_value=220, value=150, step=1
    )

    exercise_angina = st.radio(
        "Exercise-Induced Angina?",
        options=["No", "Yes"],
        horizontal=True
    )

    oldpeak = st.slider(
        "Oldpeak (ST depression)",
        min_value=0.0, max_value=6.2, value=1.0, step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        options=["Up", "Flat", "Down"],
        index=0
    )

    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# BUILD INPUT VECTOR
# ─────────────────────────────────────────
def build_input():
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Numerical
    row['Age']         = age
    row['RestingBP']   = resting_bp
    row['Cholesterol'] = cholesterol
    row['FastingBS']   = fasting_bs_code
    row['MaxHR']       = max_hr
    row['Oldpeak']     = oldpeak

    # Categorical — one-hot (drop_first baseline = all zeros)
    if sex == "Male":
        row['Sex_M'] = 1

    if chest_pain_code == 'ATA':
        row['ChestPainType_ATA'] = 1
    elif chest_pain_code == 'NAP':
        row['ChestPainType_NAP'] = 1
    elif chest_pain_code == 'TA':
        row['ChestPainType_TA'] = 1
    # ASY = baseline (all zeros)

    if resting_ecg_code == 'Normal':
        row['RestingECG_Normal'] = 1
    elif resting_ecg_code == 'ST':
        row['RestingECG_ST'] = 1
    # LVH = baseline (all zeros)

    if exercise_angina == 'Yes':
        row['ExerciseAngina_Y'] = 1

    if st_slope == 'Flat':
        row['ST_Slope_Flat'] = 1
    elif st_slope == 'Up':
        row['ST_Slope_Up'] = 1
    # Down = baseline (all zeros)

    return pd.DataFrame([row])[FEATURE_COLUMNS]

# ─────────────────────────────────────────
# MAIN AREA — results
# ─────────────────────────────────────────
if not predict_btn:
    # Landing state
    st.markdown("### How to use this tool")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nFill in the patient's clinical details in the left panel.")
    with col2:
        st.info("**Step 2**\n\nClick **Predict Risk** to run the model.")
    with col3:
        st.info("**Step 3**\n\nReview the risk score and SHAP explanation.")

    st.divider()
    st.markdown("### About this model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", "918")
    c2.metric("Model", "XGBoost")
    c3.metric("Features", "15")
    c4.metric("Cross-val AUC", "~0.93")

else:
    # ── Run prediction ──
    input_df     = build_input()
    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[0][1]
    prediction   = int(probability >= 0.5)

    # ── Result header ──
    st.markdown("## Prediction Results")
    col_res, col_prob, col_conf = st.columns(3)

    with col_res:
        if prediction == 1:
            st.error("### ❗ Heart Disease Likely")
        else:
            st.success("### ✅ Low Risk")

    with col_prob:
        st.metric(
            label="Risk Probability",
            value=f"{probability:.1%}",
            delta=f"{'High risk' if probability >= 0.5 else 'Low risk'}"
        )

    with col_conf:
        # Confidence = distance from 0.5 boundary
        confidence = abs(probability - 0.5) * 2
        conf_label = (
            "High confidence" if confidence > 0.5
            else "Moderate confidence" if confidence > 0.25
            else "Borderline — treat with caution"
        )
        st.metric(
            label="Model Confidence",
            value=f"{max(probability, 1-probability):.1%}",
            delta=conf_label
        )

    # ── Risk gauge bar ──
    st.markdown("#### Risk Score")
    gauge_color = "#E24B4A" if probability >= 0.5 else "#1D9E75"
    st.markdown(
        f"""
        <div style="background:var(--secondary-background-color);
                    border-radius:8px;height:22px;overflow:hidden;margin-bottom:6px">
          <div style="width:{probability*100:.1f}%;background:{gauge_color};
                      height:100%;border-radius:8px;transition:width .4s">
          </div>
        </div>
        <p style="font-size:13px;color:gray;margin:0">
          0% (No risk) ──────────────────────── 100% (Certain risk)
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ── Two columns: input summary + SHAP ──
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("#### Patient Summary")
        summary = {
            "Age":              age,
            "Sex":              sex,
            "Chest Pain":       chest_pain_code,
            "Resting BP":       f"{resting_bp} mmHg",
            "Cholesterol":      f"{cholesterol} mg/dL",
            "Fasting BS > 120": "Yes" if fasting_bs_code else "No",
            "Resting ECG":      resting_ecg_code,
            "Max Heart Rate":   max_hr,
            "Exercise Angina":  exercise_angina,
            "Oldpeak":          oldpeak,
            "ST Slope":         st_slope,
        }
        for k, v in summary.items():
            col_k, col_v = st.columns([1.2, 1])
            col_k.markdown(f"<span style='color:gray;font-size:13px'>{k}</span>",
                           unsafe_allow_html=True)
            col_v.markdown(f"<span style='font-size:13px;font-weight:500'>{v}</span>",
                           unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Why did the model predict this?")
        st.caption("SHAP values show which features pushed the risk up (red) or down (blue)")

        try:
            explainer   = shap.TreeExplainer(model)
            input_named = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
            shap_vals   = explainer.shap_values(input_named)

            fig, ax = plt.subplots(figsize=(7, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_vals[0],
                    base_values   = explainer.expected_value,
                    data          = input_named.iloc[0],
                    feature_names = FEATURE_COLUMNS
                ),
                show=False,
                max_display=10
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    st.divider()

    # ── Disclaimer ──
    st.caption(
        "⚠️ This tool is for research and educational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )
