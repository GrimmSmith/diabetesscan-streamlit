import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesScan · AI Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #090e1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -5%, rgba(0,212,170,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 40% 40% at 90% 85%, rgba(0,100,200,0.06) 0%, transparent 60%);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 40px 0 32px;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00d4aa;
    display: inline-block;
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.2);
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #e8edf5;
    line-height: 1.15;
    margin-bottom: 12px;
}
.hero-title span { color: #00d4aa; }
.hero-sub {
    color: #6b7a99;
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #00d4aa;
    padding: 10px 0 10px;
    border-bottom: 1px solid #1e2e50;
    margin-bottom: 18px;
    margin-top: 8px;
}

/* ── Card wrappers ── */
.form-card, .result-card {
    background: #141e35;
    border: 1px solid #1e2e50;
    border-radius: 18px;
    padding: 32px 36px;
    margin-bottom: 20px;
}

/* ── Inputs styling ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #0f1729 !important;
    border: 1px solid #1e2e50 !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.12) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #9aaac0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: #00d4aa !important;
    color: #090e1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(0,212,170,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    background: #00f0c2 !important;
    box-shadow: 0 8px 32px rgba(0,212,170,0.38) !important;
    transform: translateY(-1px) !important;
}

/* ── Result cards ── */
.result-diabetic {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.28);
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.result-safe {
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.28);
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.result-moderate {
    background: rgba(255,209,102,0.08);
    border: 1px solid rgba(255,209,102,0.28);
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.result-icon { font-size: 3.5rem; margin-bottom: 12px; }
.result-risk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e8edf5;
    margin-bottom: 8px;
}
.result-sub { color: #6b7a99; font-size: 0.92rem; line-height: 1.6; }

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 14px; margin: 20px 0; }
.metric-box {
    flex: 1;
    background: #141e35;
    border: 1px solid #1e2e50;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #00d4aa;
    margin-bottom: 4px;
}
.metric-key { font-size: 0.75rem; color: #6b7a99; font-weight: 500; }

/* ── Summary table ── */
.summ-table { width: 100%; border-collapse: collapse; }
.summ-table td {
    padding: 9px 12px;
    font-size: 0.88rem;
    border-bottom: 1px solid #1a2540;
    color: #e8edf5;
}
.summ-table td:first-child { color: #6b7a99; font-weight: 500; width: 50%; }
.summ-table tr:last-child td { border-bottom: none; }

/* ── Footer ── */
.footer-note {
    text-align: center;
    margin-top: 16px;
    font-size: 0.76rem;
    color: #4a5a78;
    padding: 16px;
}

/* Divider */
.divider { border: none; border-top: 1px solid #1e2e50; margin: 22px 0; }

/* Progress bar override */
div[data-testid="stProgress"] > div > div > div {
    background: #00d4aa !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("best_model.pkl")
    return bundle

bundle       = load_model()
model        = bundle["model"]
scaler       = bundle["scaler"]
imputer      = bundle["imputer"]
imputer_cols = bundle["imputer_cols"]
feature_cols = bundle["feature_cols"]

BMI_CATEGORY  = {"Normal": 0, "Obese": 1, "Overweight": 2, "Underweight": 3}
AGE_GROUP     = {"Adult": 0, "Mid": 1, "Senior": 2, "Young": 3}
GLUCOSE_LEVEL = {"Diabetes": 0, "Normal": 1, "Prediabetes": 2}


def predict(patient: dict) -> dict:
    df = pd.DataFrame([patient])
    df[imputer_cols]            = imputer.transform(df[imputer_cols])
    df["Glucose_BMI"]           = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / (df["Glucose"] + 1e-5)
    df["Age_Preg_Risk"]         = df["Age"] * df["Pregnancies"]
    df = df.reindex(columns=feature_cols)
    df_sc = scaler.transform(df)
    pred  = model.predict(df_sc)[0]
    prob  = model.predict_proba(df_sc)[0][1] * 100
    if prob > 60:
        risk, risk_class = "High Risk", "result-diabetic"
        risk_color = "#ff4d6d"
    elif prob > 40:
        risk, risk_class = "Moderate Risk", "result-moderate"
        risk_color = "#ffd166"
    else:
        risk, risk_class = "Low Risk", "result-safe"
        risk_color = "#00d4aa"
    return {
        "diabetic": bool(pred), "probability": round(prob, 1),
        "risk": risk, "risk_class": risk_class, "risk_color": risk_color,
    }


# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">🩺 AI-Powered Clinical Tool</div>
    <div class="hero-title">Predict <span>Diabetes Risk</span><br/>in Seconds</div>
    <div class="hero-sub">Enter patient clinical data below. Our trained Gradient Boosting model
    will instantly assess diabetes probability.</div>
</div>
""", unsafe_allow_html=True)

# ── Form ────────────────────────────────────────────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)

# Section 1
st.markdown('<div class="section-label">01 — Patient Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=45.0, step=1.0)
with c2:
    pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
with c3:
    age_group = st.selectbox("Age Group", ["Young", "Adult", "Mid", "Senior"], index=2,
                            help="Young <25 · Adult 25–40 · Mid 40–55 · Senior 55+")

st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

# Section 2
st.markdown('<div class="section-label">02 — Blood & Metabolic Markers</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=138.0, step=0.1)
with c5:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=900.0, value=120.0, step=0.1)
with c6:
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=0.1)

c7, c8, c9 = st.columns(3)
with c7:
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=80.0, value=33.6, step=0.1)
with c8:
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
with c9:
    dpf = st.number_input("Pedigree Function", min_value=0.0, max_value=3.0, value=0.627, step=0.001,
                        help="Genetic diabetes risk score (0 – 2.5)")

st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

# Section 3
st.markdown('<div class="section-label">03 — Clinical Classification</div>', unsafe_allow_html=True)
c10, c11 = st.columns(2)
with c10:
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"], index=2)
with c11:
    glucose_level = st.selectbox("Glucose Level", ["Normal", "Prediabetes", "Diabetes"], index=1)

st.markdown("</div>", unsafe_allow_html=True)

# ── Submit ──────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 2])
with col_btn:
    run = st.button("🔍  Run Prediction")

# ── Result ──────────────────────────────────────────────────────
if run:
    patient_data = {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
        "BMI_Category": BMI_CATEGORY[bmi_category],
        "Age_Group":    AGE_GROUP[age_group],
        "Glucose_Level": GLUCOSE_LEVEL[glucose_level],
    }

    with st.spinner("Analysing patient data…"):
        time.sleep(0.8)
        result = predict(patient_data)

    st.markdown("---")

    # ── Verdict card ──
    icon  = "⚠️" if result["diabetic"] else "✅"
    title = "Diabetes Detected" if result["diabetic"] else "No Diabetes Detected"
    sub   = ("The model predicts a positive diabetes outcome. Please consult a physician."
            if result["diabetic"] else
            "The model predicts no diabetes. Continue monitoring health indicators.")

    st.markdown(f"""
    <div class="{result['risk_class']}">
        <div class="result-icon">{icon}</div>
        <div class="result-risk-label" style="color:{result['risk_color']}">{result['risk']}</div>
        <div class="result-title">{title}</div>
        <div class="result-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar ──
    st.markdown("<br>", unsafe_allow_html=True)
    lcol, rcol = st.columns([4, 1])
    with lcol:
        st.markdown(f"<p style='color:#6b7a99;font-size:0.82rem;margin-bottom:6px'>Diabetes Probability</p>",
                    unsafe_allow_html=True)
        st.progress(int(result["probability"]))
    with rcol:
        st.markdown(f"""
        <div style='text-align:right;padding-top:18px;
            font-family:JetBrains Mono,monospace;font-size:1.6rem;
            font-weight:600;color:{result['risk_color']}'>
            {result['probability']}%
        </div>""", unsafe_allow_html=True)

    # ── Metric boxes ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val">{result['probability']}%</div>
            <div class="metric-key">Probability Score</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:{result['risk_color']}">{result['risk'].split()[0]}</div>
            <div class="metric-key">Risk Category</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">GB</div>
            <div class="metric-key">Model Used</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{'Yes' if result['diabetic'] else 'No'}</div>
            <div class="metric-key">Outcome</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input summary ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient Input Summary</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    fields_left = [
        ("Age", f"{age} yrs"), ("Pregnancies", int(pregnancies)),
        ("Age Group", age_group), ("Glucose", f"{glucose} mg/dL"),
        ("Insulin", f"{insulin} µU/mL"), ("Blood Pressure", f"{blood_pressure} mmHg"),
    ]
    fields_right = [
        ("BMI", f"{bmi} kg/m²"), ("Skin Thickness", f"{skin_thickness} mm"),
        ("Pedigree Function", dpf), ("BMI Category", bmi_category),
        ("Glucose Level", glucose_level), ("Age Group", age_group),
    ]

    with col_a:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields_left)
        st.markdown(f'<table class="summ-table">{rows}</table>', unsafe_allow_html=True)
    with col_b:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields_right)
        st.markdown(f'<table class="summ-table">{rows}</table>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    ⚕ For educational & clinical reference only. Not a substitute for professional medical diagnosis.<br/>
    <span style="color:#2a3a5a">Maximus · Diploma in CSE 2026 · Jamshedpur</span>
</div>
""", unsafe_allow_html=True)
