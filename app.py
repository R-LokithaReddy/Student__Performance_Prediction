import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load("student_performance_model.pkl")
features = joblib.load("model_features.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.markdown("Predict student final performance using assessment and engagement data.")

st.header("Enter Student Details")

# Numerical Inputs
score_mean = st.number_input("Average Assessment Score", 0.0, 100.0, 50.0)

score_max = st.number_input("Maximum Score", 0.0, 100.0, 80.0)

score_min = st.number_input("Minimum Score", 0.0, 100.0, 40.0)

score_std = st.number_input("Score Standard Deviation", 0.0, 100.0, 10.0)

total_assessments = st.number_input(
    "Total Assessments",
    min_value=0,
    value=5
)

is_banked_mean = st.number_input(
    "Banked Assessment Mean",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

clicks_total = st.number_input(
    "Total VLE Clicks",
    min_value=0,
    value=100
)

vle_mean = st.number_input(
    "Average VLE Date",
    value=50.0
)

vle_max = st.number_input(
    "Maximum VLE Date",
    value=100.0
)

vle_min = st.number_input(
    "Minimum VLE Date",
    value=10.0
)

vle_activity_range = vle_max - vle_min

engagement_score = clicks_total * score_mean

# Categorical Inputs
gender = st.selectbox(
    "Gender",
    ["M", "F"]
)

region = st.text_input(
    "Region",
    "East Anglian Region"
)

highest_education = st.selectbox(
    "Highest Education",
    [
        "A Level or Equivalent",
        "Lower Than A Level",
        "HE Qualification",
        "Post Graduate Qualification"
    ]
)

imd_band = st.selectbox(
    "IMD Band",
    [
        "0-10%",
        "10-20",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%"
    ]
)

age_band = st.selectbox(
    "Age Band",
    [
        "0-35",
        "35-55",
        "55<="
    ]
)

num_of_prev_attempts = st.number_input(
    "Previous Attempts",
    min_value=0,
    value=0
)

studied_credits = st.number_input(
    "Studied Credits",
    min_value=0,
    value=60
)

disability = st.selectbox(
    "Disability",
    ["N", "Y"]
)

if st.button("Predict Performance"):

    input_dict = {
        "score_mean": score_mean,
        "score_max": score_max,
        "score_min": score_min,
        "score_std": score_std,
        "total_assessments": total_assessments,
        "is_banked_mean": is_banked_mean,
        "clicks_total": clicks_total,
        "vle_mean": vle_mean,
        "vle_max": vle_max,
        "vle_min": vle_min,
        "vle_activity_range": vle_activity_range,
        "engagement_score": engagement_score,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability
    }

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(input_df)

    result = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Result: {result}")

    if result.lower() == "distinction":
        st.balloons()