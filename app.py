import numpy as np
import streamlit as st
from keras.models import load_model
from sklearn import preprocessing
from datetime import datetime

# Load the pre-trained model
try:
    model = load_model("best_model.keras")
    #st.success("Model loaded successfully.")
except FileNotFoundError:
    st.error("Saved model not found. Please ensure the model file is in the directory.")
    st.stop()

# User interface for inputs
st.title("Hridaya Netra : AI-powered Heart Stroke Predictor")

# Input fields for user data
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

# Standardize the user input
s_scaler = preprocessing.StandardScaler()
user_data = s_scaler.fit_transform(user_data)  # Using the same scaler used for training

# Predict and display the report in a "popup" modal
if st.button("Predict"):
    prediction = model.predict(user_data)[0][0]  # Get single value prediction
    risk_percentage = round(prediction * 100, 1)  # Convert to percentage

    # Determine risk category based on percentage
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

    # Simulated Popup (Modal Window)
    with st.expander("🔍 View Your Personalized CVD Risk Report", expanded=True):
        st.markdown("## Patient Advice")
        st.markdown("""
        ### What is CVD Risk?
        CVD risk means your risk of a fatal or non-fatal cardiovascular disease event (a composite of cardiovascular mortality, non-fatal myocardial infarction, and non-fatal stroke) in the next 10 years.
        """)

        st.markdown("## Your Results")
        st.markdown(f"**Examination Date:** {datetime.now().strftime('%d %B %Y')}")
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Sex:** {'Male' if sex == 1 else 'Female'}")
        st.markdown(f"**Your 10-year risk of fatal and non-fatal CVD events is:****{risk_percentage}%**")

    # Second expandable modal for advice
    with st.expander("📝 Patient's Advice Based on Risk Category", expanded=True):
        st.markdown(f"## Risk Category: **{risk_category}**")
        st.markdown(f"**Your Advice:** {advice}")
        st.markdown("""
        ### Healthy Lifestyle Tips:
        - Engage in **150 - 300 minutes of moderate-intensity** or **75 - 150 minutes of vigorous-intensity** aerobic activity weekly.
        - Follow a **Mediterranean diet**: consume healthy fats, fruits, vegetables, and whole grains.
        - Avoid smoking and limit alcohol consumption.
        - Maintain a healthy body weight and monitor your blood pressure regularly.
        - Manage stress through meditation, yoga, or other relaxation techniques.
        """)

        if risk_category in ["High", "Very High"]:
            st.markdown("### Additional Recommendations:")
            st.markdown("""
            - Regularly consult a healthcare provider for cardiovascular monitoring.
            - Consider medication for managing cholesterol and blood pressure, if advised by your doctor.
            - Monitor blood sugar levels if you have diabetes or are at risk for it.
            """)
