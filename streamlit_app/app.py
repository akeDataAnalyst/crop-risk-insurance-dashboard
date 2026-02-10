#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Crop Risk & Payout Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Crop Risk & Insurance Payout Predictor")
st.markdown("""
Interactive tool for predicting crop failure risk and estimated insurance payout for smallholder farmers.  
Built with a strong Random Forest classifier (89.7% accuracy).
""")

# ────────────────────────────────────────────────────────────────
# Load models & label encoder
# ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        rf_clf = joblib.load("../models/risk_classifier_rf_v3.joblib")
        le = joblib.load("../models/risk_class_encoder.joblib")
        return rf_clf, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

rf_clf, le = load_model()

# ────────────────────────────────────────────────────────────────
# Sidebar inputs
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Season Parameters")
    
    country = st.selectbox("Country", ["Kenya", "Ethiopia", "Uganda", "Zambia", "Tanzania", "Malawi"])
    crop = st.selectbox("Crop", ["Maize", "Sorghum", "Millet", "Beans", "Cassava", "Groundnut"])
    
    rainfall = st.slider("Seasonal Rainfall (mm)", 40, 2800, 600, step=10)
    temp = st.slider("Avg Temperature (°C)", 15.0, 36.0, 25.0, step=0.5)
    ndvi = st.slider("NDVI Peak", 0.04, 0.96, 0.60, step=0.01)
    soil_ph = st.slider("Soil pH", 4.3, 8.4, 5.9, step=0.1)
    soc = st.slider("Soil Organic Carbon (%)", 0.1, 4.0, 1.0, step=0.1)
    fertilizer = st.slider("Fertilizer N (kg/ha)", 0, 250, 50, step=5)
    pest = st.slider("Pest/Disease Level (0–3)", 0, 3, 1)
    irrigated = st.checkbox("Irrigated?", value=False)

# ────────────────────────────────────────────────────────────────
# Prediction logic
# ────────────────────────────────────────────────────────────────
if st.sidebar.button("Generate Prediction", type="primary", use_container_width=True):
    # Build input dictionary
    input_dict = {
        'rainfall_mm': rainfall,
        'avg_temp_c': temp,
        'heat_stress_days': max(0, int((temp - 28) * 4)),
        'ndvi_peak': ndvi,
        'soil_ph': soil_ph,
        'soc_percent': soc,
        'fertilizer_n_kg_ha': fertilizer,
        'pest_disease_level': pest,
        'irrigated': 1 if irrigated else 0
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode country & crop
    country_cols = ['country_Kenya', 'country_Malawi', 'country_Tanzania', 'country_Uganda', 'country_Zambia']
    crop_cols = ['crop_Cassava', 'crop_Groundnut', 'crop_Maize', 'crop_Millet', 'crop_Sorghum']

    for c in ["Kenya", "Malawi", "Tanzania", "Uganda", "Zambia"]:
        input_df[f'country_{c}'] = 1 if country == c else 0

    for cr in ["Cassava", "Groundnut", "Maize", "Millet", "Sorghum"]:
        input_df[f'crop_{cr}'] = 1 if crop == cr else 0

    # Fill missing columns with 0
    expected_cols = [
        'rainfall_mm', 'avg_temp_c', 'heat_stress_days', 'ndvi_peak', 'soil_ph',
        'soc_percent', 'fertilizer_n_kg_ha', 'pest_disease_level', 'irrigated',
        'country_Kenya', 'country_Malawi', 'country_Tanzania', 'country_Uganda', 'country_Zambia',
        'crop_Cassava', 'crop_Groundnut', 'crop_Maize', 'crop_Millet', 'crop_Sorghum'
    ]

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # Predict
    risk_encoded = rf_clf.predict(input_df)[0]
    risk_level = le.inverse_transform([risk_encoded])[0]

    # Approximate payout based on risk level
    if risk_level == 'Low':
        payout_est = 50
        payout_range = "$0 – $100 (minimal/no payout)"
    elif risk_level == 'Medium':
        payout_est = 250
        payout_range = "$150 – $350 (partial loss)"
    else:  # High
        payout_est = 500
        payout_range = "$400 – $600 (severe loss)"

    # Display results
    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Risk Level", risk_level, 
                delta_color="normal" if risk_level == "Low" else "inverse" if risk_level == "High" else None)
    col2.metric("Estimated Payout (USD/ha)", f"${payout_est}", delta_color="normal")
    col3.metric("Payout Range", payout_range)

    st.success("Model performance: 89.7% overall accuracy | Medium class recall: 71.7%")
    st.info("Payout is approximated from predicted risk class (High = severe loss, Medium = partial, Low = minimal/no payout).")

# ────────────────────────────────────────────────────────────────
# Footer / Caption (added as requested)
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666666; font-size: 0.9em;'>"
    "Developed by Aklilu Abera"
    "</p>",
    unsafe_allow_html=True
)

