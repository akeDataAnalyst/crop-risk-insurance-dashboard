Crop Risk & Insurance Payout Prediction Dashboard | Python, Streamlit, Random Forest
Built an end-to-end AI-powered dashboard predicting crop failure risk levels and estimated insurance payouts for smallholder farmers, enabling data-driven decisions for agricultural insurers and field officers.
•	Engineered features from weather (rainfall, temperature, heat stress), vegetation (NDVI), soil, and management data to model seasonal loss probability.
•	Trained Random Forest classifier achieving 86.2% overall accuracy and 80.0% recall on minority Medium-risk class (High-risk F1: 0.935).
•	Delivered an interactive Streamlit application allowing users to input season parameters and receive instant risk classification and payout range estimation.
•	Project simulates real-world index-based crop insurance workflows (inspired by Pula’s mission to protect uninsured smallholders).
•	https://github.com/akeDataAnalyst/crop-risk-insurance-dashboard

-------------------------------------

# Crop Risk & Insurance Payout Prediction Dashboard

**AI-Powered Crop Risk Assessment & Insurance Payout Predictor**  
An interactive Streamlit web application that predicts crop failure risk (Low/Medium/High) and estimates insurance payout for smallholder farmers using weather, soil, vegetation (NDVI), and agronomic data.

Built to support agricultural insurers and field officers in making faster, data-driven decisions — inspired by Pula's mission to protect uninsured smallholders.

[![Streamlit App](https://img.shields.io/badge/Launch%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-streamlit-cloud-link-here)  

## Project Overview

This project simulates real-world index-based crop insurance workflows by:
- Predicting risk level (Low/Medium/High) with a Random Forest classifier (86.2% overall accuracy, 80.0% recall on minority Medium class)
- Estimating payout (USD/ha) via rule-based approximation derived from predicted risk
- Providing an interactive dashboard for field-friendly input and instant results

### Key Features
- User inputs: country, crop, seasonal rainfall, temperature, NDVI, soil pH, SOC, fertilizer, pest level, irrigation
- Instant prediction of risk category and estimated payout range
- Strong performance on high-risk detection (F1 0.935 for High class)
- Small, efficient model (1.96 MB) suitable for deployment

## Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest), XGBoost (explored)
- **Dashboard**: Streamlit
- **Model Saving**: joblib
- **Visualization**: matplotlib, seaborn (in notebooks)
