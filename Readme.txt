Crop Risk & Insurance Payout Prediction Dashboard | Python, Streamlit, XGBoost, Random Forest
Developed an AI-powered interactive web application that predicts crop failure risk levels and estimates insurance payouts for smallholder farmers, enabling faster, data-driven decisions for agricultural insurers and field teams.
•	Engineered features from weather (rainfall, temperature, heat stress), satellite vegetation (NDVI), soil, and management data to capture key risk drivers.
•	Trained a Random Forest classifier achieving 89.7% overall accuracy and 71.7% recall on the minority Medium-risk class (High-risk F1: 0.949).
•	Designed a user-friendly Streamlit dashboard allowing input of season parameters and instant visualization of predicted risk level and estimated payout range.
•	Aligned solution with real-world index-based crop insurance workflows (inspired by Pula’s mission to protect uninsured smallholders).
•	https://github.com/[your-username]/crop-risk-insurance-dashboard



-------------------------------------

# Crop Risk & Insurance Payout Prediction Dashboard

**AI-Powered Crop Risk Assessment & Insurance Payout Predictor**  
An interactive Streamlit web application that predicts crop failure risk (Low/Medium/High) and estimates insurance payout for smallholder farmers using weather, soil, vegetation (NDVI), and agronomic data.

Built to support agricultural insurers and field officers in making faster, data-driven decisions — inspired by Pula's mission to protect uninsured smallholders.

[![Streamlit App](https://img.shields.io/badge/Launch%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-streamlit-cloud-link)  


## Project Overview

This project simulates real-world index-based crop insurance workflows by:
- Predicting risk level (Low/Medium/High) with a Random Forest classifier (89.7% overall accuracy, 71.7% recall on minority Medium class)
- Estimating payout (USD/ha) via rule-based approximation derived from predicted risk
- Providing an interactive dashboard for field-friendly input and instant results

### Key Features
- User inputs: country, crop, rainfall, temperature, NDVI, soil pH, SOC, fertilizer, pest level, irrigation
- Instant prediction of risk category and estimated payout range
- Strong performance on high-risk detection (F1 0.949 for High class)

## Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest), XGBoost
- **Dashboard**: Streamlit
- **Model Saving**: joblib
- **Visualization**: matplotlib, seaborn (in notebooks)