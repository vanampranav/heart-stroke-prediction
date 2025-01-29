import numpy as np
import streamlit as st
import joblib
from keras.models import load_model
from sklearn import preprocessing
from datetime import datetime

# Load pre-trained models
try:
    nn_model = load_model("heart_stroke_model.h5")  # Neural Network
    rf_model = joblib.load("random_forest_model.pkl")  # Random Forest
    st.success("Models loaded successfully.")
except FileNotFoundError:
    st.error("Saved models not found. Please ensure the model files are in the directory.")
    st.stop()

# Title
st.title("🫀 Hridaya Netra: AI-powered Heart Stroke Predictor")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=60)
anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, max_value=5000, value=100)
diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=10, max_value=80, value=35)
high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=137)
sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
time = st.number_input("Follow-up Period (days)", min_value=1, max_value=300, value=120)

# Prepare input for prediction
user_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                       high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                       smoking, time]])

# Standardize input (Same scaler used for training)
scaler = preprocessing.StandardScaler()
user_data = scaler.fit_transform(user_data)

# Prediction
if st.button("🔍 Predict"):
    # Feature extraction using Neural Network
    features = nn_model.predict(user_data)

    # Final prediction using Random Forest
    prediction = rf_model.predict(features)[0]

    # Convert prediction to risk percentage
    risk_percentage = round(prediction * 100, 1)

    # Determine risk category
    if risk_percentage < 2.5:
        risk_category = "Low"
        advice = "Your risk is low. Maintain a healthy lifestyle to keep your risk low."
    elif 2.5 <= risk_percentage < 7.5:
        risk_category = "Moderate"
        advice = "Your risk is moderate. Adopt lifestyle changes and monitor your health regularly."
    elif 7.5 <= risk_percentage < 20:
        risk_category = "High"
        advice = "Your risk is high. Consult a healthcare professional and consider lifestyle or medical interventions."
    else:
        risk_category = "Very High"
        advice = "Your risk is very high. Immediate medical attention and aggressive management are recommended."

    # Display results
    with st.expander("📋 View Your Personalized CVD Risk Report", expanded=True):
        st.markdown("## 🏥 Patient Advice")
        st.markdown("""
        ### 🫀 What is CVD Risk?
        Cardiovascular Disease (CVD) risk means your chance of experiencing a heart-related issue (such as a stroke or heart attack) in the next 10 years.
        """)
        st.markdown("## 🔎 Your Results")
        st.markdown(f"**🗓️ Examination Date:** {datetime.now().strftime('%d %B %Y')}")
        st.markdown(f"**👤 Age:** {age}")
        st.markdown(f"**⚧ Sex:** {'Male' if sex == 1 else 'Female'}")
        st.markdown(f"**📊 Your 10-year risk of a heart event is:** **{risk_percentage}%**")

    with st.expander("📌 Patient's Advice Based on Risk Category", expanded=True):
        st.markdown(f"## 🔹 Risk Category: **{risk_category}**")
        st.markdown(f"**🩺 Your Advice:** {advice}")
        st.markdown("""
        ### 🏃‍♂️ Healthy Lifestyle Tips:
        - Engage in **150 - 300 minutes of moderate-intensity** or **75 - 150 minutes of vigorous-intensity** aerobic activity weekly.
        - Follow a **Mediterranean diet**: consume healthy fats, fruits, vegetables, and whole grains.
        - Avoid smoking and limit alcohol consumption.
        - Maintain a healthy body weight and monitor your blood pressure regularly.
        - Manage stress through meditation, yoga, or relaxation techniques.
        """)

        if risk_category in ["High", "Very High"]:
            st.markdown("### 🚨 Additional Recommendations:")
            st.markdown("""
            - Consult a healthcare provider for regular cardiovascular monitoring.
            - Consider medication for managing cholesterol and blood pressure if advised by your doctor.
            - Monitor blood sugar levels if you have diabetes or are at risk for it.
            """)

