import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved artifacts
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    numeric_features = joblib.load("numeric_features.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Debugging: Display loaded features
st.write("Loaded numeric features:", numeric_features)
st.write("Loaded feature columns:", feature_columns)

st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the house details below to predict the price and see why the prediction was made (SHAP explainability).")

# ---------------------------
# User Inputs
# ---------------------------
total_sqft = st.number_input("Total Sqft", min_value=100, max_value=10000, value=1000)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
location = st.selectbox("Location", sorted([col for col in feature_columns if col not in numeric_features]))
distance_to_center_km = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=15.0)
location_avg_price_per_sqft = st.number_input("Location Avg Price per Sqft (₹)", min_value=1000, max_value=50000, value=5000)
school_dist_km = st.number_input("Distance to Nearest School (km)", min_value=0.0, max_value=20.0, value=2.0)
hospital_dist_km = st.number_input("Distance to Nearest Hospital (km)", min_value=0.0, max_value=20.0, value=3.0)
metro_dist_km = st.number_input("Distance to Nearest Metro Station (km)", min_value=0.0, max_value=20.0, value=1.5)
year = st.number_input("Year Built", min_value=1900, max_value=2025, value=2020)

# ---------------------------
# Prepare Input DataFrame
# ---------------------------
input_dict = {
    "total_sqft": total_sqft,
    "bhk": bhk,
    "bath": bath,
    "balcony": balcony,
    "distance_to_center_km": distance_to_center_km,
    "location_avg_price_per_sqft": location_avg_price_per_sqft,
    "school_dist_km": school_dist_km,
    "hospital_dist_km": hospital_dist_km,
    "metro_dist_km": metro_dist_km,
    "year": year
}

# Create DataFrame with all feature columns
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0

# Fill numeric columns
for col in input_dict:
    if col in feature_columns:
        input_df.at[0, col] = input_dict[col]

# Set location column to 1
if location in feature_columns:
    input_df.at[0, location] = 1
else:
    st.warning(f"Location '{location}' not found in training data. Using 'other' location features.")
    # Set default values for unknown location
    input_df.at[0, 'distance_to_center_km'] = 15.0
    input_df.at[0, 'location_avg_price_per_sqft'] = 5000
    input_df.at[0, 'school_dist_km'] = 2.0
    input_df.at[0, 'hospital_dist_km'] = 2.5
    input_df.at[0, 'metro_dist_km'] = 3.0

# Ensure input_df matches feature_columns order
input_df = input_df[feature_columns]
st.write("Input DataFrame columns:", input_df.columns.tolist())

# Scale only the numeric features
input_df_scaled = input_df.copy()
input_df_scaled[numeric_features] = scaler.transform(input_df[numeric_features])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Price"):
    try:
        # Predict and reverse log-transformation
        prediction = model.predict(input_df_scaled)[0]
        # Assuming price was in lakhs during training
        prediction = np.exp(prediction) * 100000  # Convert log(lakhs) to rupees
        st.success(f"🏠 Predicted House Price: ₹ {prediction:,.2f}")

        # SHAP explainability
        explainer = shap.Explainer(model, input_df_scaled)
        shap_values = explainer(input_df_scaled)
        st.subheader("Prediction Explainability")
        
        # Render SHAP waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")