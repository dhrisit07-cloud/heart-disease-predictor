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
# SIDEBAR — patient input (typed)
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Patient Details")
    st.caption("Enter the patient's clinical information below.")

    st.subheader("Numerical Values")

    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120,
        value=54, step=1,
        help="Patient age in years (1–120)"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=50, max_value=300,
        value=130, step=1,
        help="Resting blood pressure in mmHg (50–300)"
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0, max_value=700,
        value=237, step=1,
        help="Serum cholesterol in mg/dL. Enter 0 if unknown — will be replaced with dataset median (237)"
    )
    if cholesterol == 0:
        st.warning("Cholesterol 0 detected — will use dataset median: 237 mg/dL")
        cholesterol = 237

    max_hr = st.number_input(
        "Max Heart Rate Achieved",
        min_value=40, max_value=250,
        value=150, step=1,
        help="Maximum heart rate achieved during exercise (40–250)"
    )

    oldpeak = st.number_input(
        "Oldpeak (ST depression)",
        min_value=0.0, max_value=10.0,
        value=1.0, step=0.1,
        format="%.1f",
        help="ST depression induced by exercise relative to rest (0.0–10.0)"
    )

    st.subheader("Categorical Values")

    sex = st.radio(
        "Sex",
        options=["Male", "Female"],
        horizontal=True
    )

    fasting_bs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL?",
        options=["No", "Yes"],
        horizontal=True
    )
    fasting_bs_code = 1 if fasting_bs == "Yes" else 0

    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[
            "ATA — Atypical Angina",
            "NAP — Non-Anginal Pain",
            "ASY — Asymptomatic",
            "TA  — Typical Angina"
        ],
        index=2
    )
    chest_pain_code = chest_pain.split(" — ")[0].strip()

    resting_ecg = st.selectbox(
        "Resting ECG",
        options=[
            "Normal",
            "ST — ST-T wave abnormality",
            "LVH — Left ventricular hypertrophy"
        ],
        index=0
    )
    resting_ecg_code = resting_ecg.split(" — ")[0].strip()

    exercise_angina = st.radio(
        "Exercise-Induced Angina?",
        options=["No", "Yes"],
        horizontal=True
    )

    st_slope = st.selectbox(
        "ST Slope",
        options=["Up", "Flat", "Down"],
        index=0
    )

    st.divider()
    predict_btn = st.button(
        "🔍 Predict Risk",
        type="primary",
        use_container_width=True
    )

# ─────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────
def validate_inputs():
    errors = []
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120")
    if resting_bp < 50 or resting_bp > 300:
        errors.append("Resting BP must be between 50 and 300 mmHg")
    if max_hr < 40 or max_hr > 250:
        errors.append("Max Heart Rate must be between 40 and 250")
    if oldpeak < 0.0 or oldpeak > 10.0:
        errors.append("Oldpeak must be between 0.0 and 10.0")
    return errors

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
# MAIN AREA
# ─────────────────────────────────────────
if not predict_btn:
    st.markdown("### How to use this tool")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nType the patient's clinical values in the left panel.")
    with col2:
        st.info("**Step 2**\n\nSelect categorical values from the dropdowns.")
    with col3:
        st.info("**Step 3**\n\nClick **Predict Risk** to see the result and SHAP explanation.")

    st.divider()
    st.markdown("### About this model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", "918")
    c2.metric("Model", "XGBoost")
    c3.metric("Test Accuracy", "86.41%")
    c4.metric("Cross-val AUC", "0.9164")

else:
    # Validate first
    errors = validate_inputs()
    if errors:
        for e in errors:
            st.error(f"Input error: {e}")
        st.stop()

    # Run prediction
    input_df     = build_input()
    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[0][1]
    prediction   = int(probability >= 0.5)
    confidence   = max(probability, 1 - probability)
    conf_label   = (
        "High confidence" if confidence > 0.75
        else "Moderate confidence" if confidence > 0.60
        else "Borderline — treat with caution"
    )

    # Result header
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
            delta="High risk" if probability >= 0.5 else "Low risk"
        )

    with col_conf:
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.1%}",
            delta=conf_label
        )

    # Risk gauge bar
    st.markdown("#### Risk Score")
    gauge_color = "#E24B4A" if probability >= 0.5 else "#1D9E75"
    pct = probability * 100
    label_inside = pct > 20
    inside_label = f"<span style="color:white;font-size:13px;font-weight:600;padding-right:10px">{pct:.1f}%</span>" if label_inside else ""
    outside_label = f"<span style="position:absolute;left:calc({pct:.1f}% + 8px);top:50%;transform:translateY(-50%);font-size:13px;font-weight:600">{pct:.1f}%</span>" if not label_inside else ""
    st.markdown(
        f"""
        <div style="position:relative;background:var(--secondary-background-color);
                    border-radius:8px;height:32px;margin-bottom:8px">
          <div style="width:{pct:.1f}%;background:{gauge_color};height:100%;
                      border-radius:8px;display:flex;align-items:center;
                      justify-content:flex-end;min-width:0px">
            {inside_label}
          </div>
          {outside_label}
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:gray;margin-top:2px">
          <span>0% — No risk</span><span>50% — Borderline</span><span>100% — Certain risk</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Two columns: summary + SHAP
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("#### Patient Summary")
        summary = {
            "Age":              f"{age} years",
            "Sex":              sex,
            "Chest Pain":       chest_pain_code,
            "Resting BP":       f"{resting_bp} mmHg",
            "Cholesterol":      f"{cholesterol} mg/dL",
            "Fasting BS > 120": fasting_bs,
            "Resting ECG":      resting_ecg_code,
            "Max Heart Rate":   f"{max_hr} bpm",
            "Exercise Angina":  exercise_angina,
            "Oldpeak":          f"{oldpeak:.1f}",
            "ST Slope":         st_slope,
        }
        for k, v in summary.items():
            ck, cv = st.columns([1.2, 1])
            ck.markdown(
                f"<span style='color:gray;font-size:13px'>{k}</span>",
                unsafe_allow_html=True
            )
            cv.markdown(
                f"<span style='font-size:13px;font-weight:500'>{v}</span>",
                unsafe_allow_html=True
            )

    with col_right:
        st.markdown("#### Why did the model predict this?")
        st.caption("SHAP values — red bars push risk up, blue bars push risk down")

        try:
            explainer   = shap.TreeExplainer(model)
            input_named = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
            shap_vals   = explainer.shap_values(input_named)

            fig, ax = plt.subplots(figsize=(7, 4))
            shap.plots.waterfall(
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
    st.caption(
        "⚠️ This tool is for research and educational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )
