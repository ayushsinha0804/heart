import streamlit as st
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #1a1a2e;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9ca3af;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f3f4f6;
    }

    .result-safe {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1.5px solid #6ee7b7;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-risk {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1.5px solid #fca5a5;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }

    .result-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
    }

    .confidence-bar-container {
        background: #f3f4f6;
        border-radius: 99px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .stButton > button {
        background: #1a1a2e !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        width: 100%;
        transition: all 0.2s ease;
        font-family: 'DM Sans', sans-serif !important;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        background: #16213e !important;
        transform: translateY(-1px);
    }

    .disclaimer {
        font-size: 0.78rem;
        color: #9ca3af;
        text-align: center;
        margin-top: 1.5rem;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 10px;
        line-height: 1.6;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    # Prefer a saved sklearn Pipeline (preprocessing + scaler + classifier)
    pipeline_path = 'models/pipeline.pkl'
    features_path = 'models/feature_columns.pkl'

    if os.path.exists(pipeline_path) and os.path.exists(features_path):
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        return pipeline, feature_columns

    # Fallback: old behavior (not used in this workspace)
    return None, None


pipeline, feature_columns = load_or_train_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Heart Disease<br>Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your clinical parameters below to assess cardiovascular risk.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Model pipeline not found. Please run `model_training.py` to generate models/pipeline.pkl first.")
    st.stop()

# ── Form ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
with col2:
    sex = st.selectbox("Biological Sex", options=["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

st.markdown('<div class="section-header">Symptoms & Clinical Findings</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ]
    )
    cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)

with col4:
    exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0

col5, col6 = st.columns(2)
with col5:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120, step=1)
with col6:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)

st.markdown('<div class="section-header">Diagnostic Results</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    fbs_val = 1 if fbs == "Yes" else 0
with col8:
    restecg = st.selectbox(
        "Resting ECG Results",
        options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

col9, col10 = st.columns(2)
with col9:
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150, step=1)
with col10:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")

col11, col12, col13 = st.columns(3)
with col11:
    slope = st.selectbox("Slope of Peak ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
with col12:
    ca = st.selectbox("Major Vessels Colored (0–3)", options=[0, 1, 2, 3])
with col13:
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
    thal_val = [0, 1, 2].index(["Normal", "Fixed Defect", "Reversible Defect"].index(thal)) + 1

# ── Predict ──────────────────────────────────────────────────────────────────
# Build dynamic form inputs based on feature_columns
st.markdown('<div class="section-header">Model Inputs</div>', unsafe_allow_html=True)
df_sample = None
if os.path.exists('heart_disease_data2.csv'):
    df_sample = pd.read_csv('heart_disease_data2.csv')

user_inputs = {}
numeric_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
binary_features = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex']

for col in feature_columns:
    if col in numeric_features:
        default = 0.0
        if df_sample is not None and col in df_sample.columns:
            default = float(pd.to_numeric(df_sample[col], errors='coerce').median())
        user_inputs[col] = st.number_input(col, value=default)
    elif col in binary_features:
        # present Yes/No for binaries, Sex uses Male/Female
        if col == 'Sex':
            user_inputs[col] = st.selectbox('Sex', options=['Male', 'Female'])
        else:
            user_inputs[col] = st.selectbox(col, options=['Yes', 'No'])
    else:
        # categorical: provide choices from sample if available
        if df_sample is not None and col in df_sample.columns:
            options = sorted(df_sample[col].dropna().unique().tolist())
            user_inputs[col] = st.selectbox(col, options=options)
        else:
            user_inputs[col] = st.text_input(col, value='')

if st.button("Analyse My Risk →"):
    # Build single-row DataFrame from user_inputs
    input_df = pd.DataFrame([{k: (v if v != '' else np.nan) for k, v in user_inputs.items()}])

    # For Sex and binary fields, keep values as strings/Yes/No which the pipeline maps
    # For Sex selectbox we used 'Male'/'Female' already
    proba = pipeline.predict_proba(input_df)[0]
    prediction = pipeline.predict(input_df)[0]
    confidence = proba[prediction] * 100

    if prediction == 0:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">💚</div>
            <div class="result-title" style="color: #065f46;">Low Risk Detected</div>
            <p style="color: #047857; font-size: 1rem; margin: 0.5rem 0 0;">
                The model does not detect significant indicators of heart disease based on the provided parameters.
            </p>
            <p style="font-size: 0.85rem; color: #6b7280; margin-top: 0.8rem;">
                Model confidence: <strong>{confidence:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-risk">
            <div class="result-icon">⚠️</div>
            <div class="result-title" style="color: #991b1b;">Elevated Risk Detected</div>
            <p style="color: #b91c1c; font-size: 1rem; margin: 0.5rem 0 0;">
                The model has detected indicators associated with heart disease. Please consult a cardiologist promptly.
            </p>
            <p style="font-size: 0.85rem; color: #6b7280; margin-top: 0.8rem;">
                Model confidence: <strong>{confidence:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Risk breakdown
    st.markdown('<div class="section-header">Risk Probability Breakdown</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("No Disease", f"{proba[0]*100:.1f}%")
    with col_b:
        st.metric("Heart Disease", f"{proba[1]*100:.1f}%")

    st.progress(float(proba[1]))

st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and is not a substitute
    for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider
    for any medical concerns.
</div>
""", unsafe_allow_html=True)
