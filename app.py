import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

# ── Color palette ───────────────────────────────────────────────
BG      = "#120a0e"
SURFACE = "#1a0e14"
ACCENT  = "#ff6eb0"
MUTED   = "#9c7a88"
TEXT    = "#f5e6ed"

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  [data-testid="stSidebar"] {{ background-color: {SURFACE}; }}

  .metric-box {{
    background: {SURFACE};
    border: 1px solid #3a1a28;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
  }}
  .metric-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {ACCENT};
  }}
  .metric-label {{
    font-size: 0.75rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .risk-high {{ color: #ff4f4f; font-size: 1.5rem; font-weight: 700; }}
  .risk-mod  {{ color: #ffb347; font-size: 1.5rem; font-weight: 700; }}
  .risk-low  {{ color: #4fff91; font-size: 1.5rem; font-weight: 700; }}

  .stButton > button {{
    background-color: {ACCENT};
    color: {BG};
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
  }}
  .stButton > button:hover {{
    background-color: #ff9cc8;
    color: {BG};
  }}

  label {{ color: {MUTED} !important; font-size: 0.8rem; letter-spacing: 0.05em; }}
  hr {{ border-color: #3a1a28; }}
</style>
""", unsafe_allow_html=True)


# ── Feature display name mapping ────────────────────────────────
FEATURE_LABELS = {
    "age":          "Age",
    "gender":       "Gender",
    "height":       "Height (cm)",
    "weight":       "Weight (kg)",
    "systolic_bp":  "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "cholesterol":  "Cholesterol Level",
    "glucose":      "Glucose Level",
    "smoker":       "Smoking Status",
    "alcohol":      "Alcohol Use",
    "active":       "Physical Activity",
}


# ── Data loading & model training ──────────────────────────────
@st.cache_data
def load_and_train():
    """
    Loads Cardio Train dataset, cleans and preprocesses it,
    trains a balanced logistic regression model.
    Cardio Train raw columns:
      id, age (days), gender (1=women,2=men), height, weight,
      ap_hi (systolic), ap_lo (diastolic), cholesterol (1/2/3),
      gluc (1/2/3), smoke, alco, active, cardio (target)
    """
    df = pd.read_csv("data/cardio_train.csv", sep=";")

    # Drop identifier — no predictive value
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Rename to readable names
    df = df.rename(columns={
        "age":        "age",
        "gender":     "gender",
        "height":     "height",
        "weight":     "weight",
        "ap_hi":      "systolic_bp",
        "ap_lo":      "diastolic_bp",
        "cholesterol":"cholesterol",
        "gluc":       "glucose",
        "smoke":      "smoker",
        "alco":       "alcohol",
        "active":     "active",
        "cardio":     "target",
    })

    # Convert age from days → years
    df["age"] = (df["age"] / 365.25).round(0).astype(int)

    # Remove physiologically impossible BP values
    df = df[(df["systolic_bp"] >= 70) & (df["systolic_bp"] <= 250)]
    df = df[(df["diastolic_bp"] >= 40) & (df["diastolic_bp"] <= 160)]

    df = df.dropna()

    features = [
        "age", "gender", "height", "weight",
        "systolic_bp", "diastolic_bp",
        "cholesterol", "glucose",
        "smoker", "alcohol", "active",
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    auc      = roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1])
    n_train  = len(df)

    return model, scaler, features, auc, n_train


model, scaler, features, auc, n_train = load_and_train()


# ── Header ──────────────────────────────────────────────────────
st.markdown("## ❤️ Cardiovascular Disease Risk Predictor")
st.markdown(
    "*Logistic regression model trained on the Cardio Train dataset "
    "and validated across Framingham Heart Study and UCI Heart Disease datasets.*"
)
st.divider()

# ── Model metrics ────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">{n_train:,}</div>'
    f'<div class="metric-label">Patients Trained On</div>'
    f'</div>',
    unsafe_allow_html=True,
)
c2.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">AUC {auc:.2f}</div>'
    f'<div class="metric-label">Model Performance</div>'
    f'</div>',
    unsafe_allow_html=True,
)
c3.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">3</div>'
    f'<div class="metric-label">Datasets Referenced</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Patient input form ───────────────────────────────────────────
st.markdown("### Enter Patient Information")

r1c1, r1c2, r1c3 = st.columns(3)
age       = r1c1.number_input("Age (years)", min_value=18, max_value=100, value=50)
gender    = r1c2.selectbox("Gender", ["Female", "Male"])
height    = r1c3.number_input("Height (cm)", min_value=100, max_value=220, value=170)

r2c1, r2c2, r2c3 = st.columns(3)
weight    = r2c1.number_input("Weight (kg)", min_value=30, max_value=200, value=75)
systolic  = r2c2.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
diastolic = r2c3.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80)

r3c1, r3c2, r3c3 = st.columns(3)
cholesterol = r3c1.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
glucose     = r3c2.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
smoker      = r3c3.selectbox("Smoker?", ["No", "Yes"])

r4c1, r4c2 = st.columns(2)
alcohol  = r4c1.selectbox("Alcohol use?", ["No", "Yes"])
active   = r4c2.selectbox("Physically active?", ["Yes", "No"])

# ── Prediction ───────────────────────────────────────────────────
if st.button("Calculate CVD Risk", type="primary"):

    # Encode inputs to match training data encoding exactly
    # gender: 1 = women, 2 = men (Cardio Train convention)
    gender_enc = 2 if gender == "Male" else 1
    chol_enc   = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
    gluc_enc   = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[glucose]
    smoke_enc  = 1 if smoker  == "Yes" else 0
    alco_enc   = 1 if alcohol == "Yes" else 0
    active_enc = 1 if active  == "Yes" else 0

    input_data = pd.DataFrame([[
        age, gender_enc, height, weight,
        systolic, diastolic,
        chol_enc, gluc_enc,
        smoke_enc, alco_enc, active_enc,
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]

    if prob > 0.60:
        risk_html = '<span class="risk-high">🔴 High Risk</span>'
        risk_note = (
            "This patient profile shows a strong CVD risk signal. "
            "Clinical evaluation and preventative intervention are strongly advised."
        )
    elif prob > 0.35:
        risk_html = '<span class="risk-mod">🟡 Moderate Risk</span>'
        risk_note = (
            "This patient profile shows an elevated CVD risk. "
            "Lifestyle modifications and monitoring are recommended."
        )
    else:
        risk_html = '<span class="risk-low">🟢 Low Risk</span>'
        risk_note = (
            "This patient profile does not show a strong CVD risk signal "
            "based on the provided inputs."
        )

    st.divider()
    st.markdown(f"### Predicted CVD Risk: {risk_html}", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED}; font-size:0.9rem;'>{risk_note}</p>",
        unsafe_allow_html=True,
    )

    # ── Probability cards ────────────────────────────────────────
    p1, p2 = st.columns(2)
    p1.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-value">{prob:.1%}</div>'
        f'<div class="metric-label">CVD Risk Probability</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    p2.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-value">{1 - prob:.1%}</div>'
        f'<div class="metric-label">No CVD Probability</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Top contributing risk factors ────────────────────────────
    st.divider()
    st.markdown("#### Top Contributing Risk Factors")
    st.caption("Based on absolute model coefficients — higher = stronger influence on prediction.")

    coefs = pd.Series(model.coef_[0], index=features).abs().sort_values(ascending=False)
    top5  = coefs.head(5)

    for feat, val in top5.items():
        label      = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        bar_width  = int((val / coefs.max()) * 100)
        st.markdown(
            f"<div style='margin-bottom:0.5rem;'>"
            f"<span style='color:{TEXT}; font-weight:600;'>{label}</span>"
            f"<span style='color:{MUTED}; font-size:0.8rem;'> — influence score: {val:.3f}</span>"
            f"<div style='background:#3a1a28; border-radius:4px; height:6px; margin-top:4px;'>"
            f"<div style='background:{ACCENT}; width:{bar_width}%; height:6px; border-radius:4px;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for portfolio demonstration only. "
        "Trained on the Cardio Train dataset (Kaggle). "
        "Not validated for clinical use. Not a substitute for medical evaluation."
    )


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        f"""
        <div style='color:{MUTED}; font-size:0.85rem; line-height:1.7;'>
        This predictor estimates cardiovascular disease risk using a 
        <b style='color:{TEXT};'>logistic regression</b> model trained on 
        70,000 patient records from the Cardio Train dataset.
        <br><br>
        <b style='color:{TEXT};'>Features used:</b> Age, gender, height, 
        weight, blood pressure, cholesterol, glucose, smoking, alcohol use, 
        and physical activity.
        <br><br>
        <b style='color:{TEXT};'>Preprocessing:</b> StandardScaler 
        normalization, outlier removal on BP values, class-balanced training.
        <br><br>
        <b style='color:{TEXT};'>Datasets referenced:</b><br>
        • Cardio Train (primary, Kaggle)<br>
        • Framingham Heart Study<br>
        • UCI Heart Disease Dataset
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"<div style='color:{MUTED}; font-size:0.75rem;'>"
        f"Built by <b style='color:{TEXT};'>TaraJee Clarke</b><br>"
        f"MS Health Informatics — Hofstra University"
        f"</div>",
        unsafe_allow_html=True,
    )
