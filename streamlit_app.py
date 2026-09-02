"""Cardiovascular risk predictor - GUI around predict_cardio_risk().

Loads the pre-trained model/scaler saved by cardiovascular_risk_model.py
(cardio_risk_model.pkl, cardio_risk_scaler.pkl) and lets a user enter their
health details in a form to get a live risk prediction.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardiovascular risk predictor",
    page_icon=":material/monitor_heart:",
    layout="centered",
)

# Column order must match X.columns from training in cardiovascular_risk_model.py
FEATURE_COLUMNS = [
    "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "age_years", "bmi",
]

LEVEL_OPTIONS = ["Normal", "Above normal", "Well above normal"]
LEVEL_TO_CODE = {label: i + 1 for i, label in enumerate(LEVEL_OPTIONS)}


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("cardio_risk_model.pkl")
    scaler = joblib.load("cardio_risk_scaler.pkl")
    return model, scaler


def predict_cardio_risk(model, scaler, *, gender, height, weight, ap_hi, ap_lo,
                         cholesterol, gluc, smoke, alco, active, age_years):
    bmi = weight / ((height / 100) ** 2)
    row = pd.DataFrame([{
        "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
        "age_years": age_years, "bmi": bmi,
    }])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]
    return prediction, probability, bmi


model, scaler = load_model_and_scaler()

st.title("Cardiovascular risk predictor")
st.caption(
    "Estimates cardiovascular disease risk from a Random Forest model "
    "trained on 70,000 patient records. Not a medical diagnosis."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        height = st.number_input("Height (cm)", min_value=100, max_value=230, value=170)
        ap_hi = st.number_input("Systolic blood pressure (upper)", min_value=60, max_value=260, value=120)
        cholesterol_label = st.segmented_control("Cholesterol level", LEVEL_OPTIONS, default="Normal")
    with col2:
        gender_label = st.segmented_control("Gender", ["Female", "Male"], default="Female")
        weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)
        ap_lo = st.number_input("Diastolic blood pressure (lower)", min_value=30, max_value=210, value=80)
        gluc_label = st.segmented_control("Glucose level", LEVEL_OPTIONS, default="Normal")

    with st.container(horizontal=True):
        smoke = st.toggle("Smoker")
        alco = st.toggle("Drinks alcohol")
        active = st.toggle("Physically active", value=True)

    submitted = st.form_submit_button("Predict risk", icon=":material/monitor_heart:", type="primary")

if submitted:
    if not cholesterol_label or not gluc_label or not gender_label:
        st.warning("Please select gender, cholesterol, and glucose level.", icon=":material/warning:")
    else:
        prediction, probability, bmi = predict_cardio_risk(
            model, scaler,
            gender=1 if gender_label == "Female" else 2,
            height=height, weight=weight,
            ap_hi=ap_hi, ap_lo=ap_lo,
            cholesterol=LEVEL_TO_CODE[cholesterol_label],
            gluc=LEVEL_TO_CODE[gluc_label],
            smoke=int(smoke), alco=int(alco), active=int(active),
            age_years=age_years,
        )

        if prediction == 1:
            st.error(f"At risk of cardiovascular disease — {probability:.1%} estimated probability",
                      icon=":material/warning:")
        else:
            st.success(f"Low risk / no disease detected — {probability:.1%} estimated probability",
                        icon=":material/check_circle:")

        col1, col2 = st.columns(2)
        col1.metric("Risk probability", f"{probability:.1%}")
        col2.metric("BMI", f"{bmi:.1f}")

        st.caption(
            "This is a statistical estimate based on historical data, not a "
            "medical diagnosis. Please consult a doctor for an actual evaluation."
        )

with st.expander("What drives this model's predictions", icon=":material/query_stats:"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])
    importance_series = pd.Series(importances, index=FEATURE_COLUMNS, name="Importance")
    importance_series = importance_series.sort_values(ascending=False)
    st.bar_chart(importance_series, horizontal=True)
    st.caption(
        "Relative importance of each input feature to the trained model, "
        f"a {type(model).__name__}."
    )
