import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
import re

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bengaluru House Price Predictor")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and other required artifacts."""
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        numeric_features = joblib.load("numeric_features.pkl")
        X_test_tree = joblib.load("X_test_tree.pkl")
        location_coords = joblib.load("location_coords.pkl")
        # Load raw data for comparisons
        raw_df = pd.read_csv("Bengaluru_House_Data.csv")
        return model, scaler, feature_columns, numeric_features, X_test_tree, raw_df, location_coords
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Please ensure the required files are in the directory.")
        st.stop()

model, scaler, feature_columns, numeric_features, X_test_tree, raw_df, location_coords = load_artifacts()

# --- Sidebar User Inputs ---
def get_user_inputs():
    """Gets user input from the sidebar."""
    st.sidebar.header("House Features")
    total_sqft = st.sidebar.number_input("Total Sqft", min_value=100, max_value=10000, value=1200)
    bhk = st.sidebar.slider("BHK", min_value=1, max_value=10, value=2)
    bath = st.sidebar.slider("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.sidebar.slider("Balconies", min_value=0, max_value=5, value=1)
    
    # Get available locations from feature columns
    locations = sorted([col for col in feature_columns if col not in numeric_features])
    location = st.sidebar.selectbox("Location", locations)

    st.sidebar.subheader("Advanced Features")
    distance_to_center_km = st.sidebar.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=15.0)
    location_avg_price_per_sqft = st.sidebar.number_input("Location Avg Price per Sqft (₹)", min_value=1000, max_value=50000, value=5000)
    school_dist_km = st.sidebar.number_input("Distance to Nearest School (km)", min_value=0.0, max_value=20.0, value=2.0)
    hospital_dist_km = st.sidebar.number_input("Distance to Nearest Hospital (km)", min_value=0.0, max_value=20.0, value=3.0)
    metro_dist_km = st.sidebar.number_input("Distance to Nearest Metro (km)", min_value=0.0, max_value=20.0, value=1.5)
    year = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2020)
    
    return {
        "total_sqft": total_sqft, "bhk": bhk, "bath": bath, "balcony": balcony, "location": location,
        "distance_to_center_km": distance_to_center_km, "location_avg_price_per_sqft": location_avg_price_per_sqft,
        "school_dist_km": school_dist_km, "hospital_dist_km": hospital_dist_km, "metro_dist_km": metro_dist_km, "year": year
    }

# --- Data Preparation ---
def prepare_input_dataframe(user_inputs):
    """Prepares the user input into a DataFrame for prediction."""
    input_df = pd.DataFrame([user_inputs])
    
    # Create a full DataFrame with all feature columns, initialized to zero
    full_df = pd.DataFrame(columns=feature_columns)
    full_df.loc[0] = 0
    
    # Populate numeric features
    for col in numeric_features:
        if col in input_df.columns:
            full_df[col] = input_df[col]

    # One-hot encode location
    if user_inputs['location'] in feature_columns:
        full_df[user_inputs['location']] = 1
    
    # Reorder columns to match training
    full_df = full_df[feature_columns]
    
    # Scale numeric features
    full_df_scaled = full_df.copy()
    full_df_scaled[numeric_features] = scaler.transform(full_df[numeric_features])
    
    return full_df_scaled, full_df

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🏡 Enhanced Bengaluru House Price Predictor")
    
    tab1, tab2, tab3 = st.tabs(["Price Predictor", "House Comparison", "Map Visualization"])

    with tab1:
        st.write("Use the sidebar to enter house details. The predicted price and feature explanations will be shown here.")
        
        user_inputs = get_user_inputs()
        
        if st.sidebar.button("Predict Price"):
            input_df_scaled, input_df_unscaled = prepare_input_dataframe(user_inputs)
            
            # --- Prediction ---
            prediction_log = model.predict(input_df_scaled)[0]
            prediction_rupees = np.exp(prediction_log) * 100000
            
            st.header("Prediction")
            st.success(f"**Predicted House Price: ₹ {prediction_rupees:,.2f}**")
            
            # --- SHAP Explanation ---
            st.header("Why this price? (Prediction Explanation)")
            explainer = shap.Explainer(model, input_df_scaled)
            shap_values = explainer(input_df_scaled)
            
            # Waterfall plot for the specific prediction
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
            
            st.info("""
                **How to read this chart:**
                - The final predicted price (in log scale) is at the top.
                - Features in **red** are pushing the price **higher**.
                - Features in **blue** are pushing the price **lower**.
                - The `E[f(X)]` is the baseline prediction before any features were considered.
            """)

            # --- Download Report ---
            report_content = generate_report_content(user_inputs, prediction_rupees, shap_values)
            st.download_button(
                label="Download Prediction Report",
                data=report_content,
                file_name="bengaluru_house_price_prediction_report.txt",
                mime="text/plain"
            )

        # --- Global SHAP Explanation ---
        with st.expander("View Global Feature Importance"):
            display_global_shap(model, X_test_tree)
        
        # --- Recommendations ---
        if st.sidebar.button("Get Recommendations", key="rec_button"):
            location_stats_df = get_location_stats(raw_df, location_coords)
            display_recommendations(user_inputs['location'], location_stats_df)


    with tab2:
        display_comparison_plots(raw_df)
    
    with tab3:
        location_stats_df = get_location_stats(raw_df, location_coords)
        display_map_visualization(location_stats_df)


def display_global_shap(model, data):
    """Calculates and displays the global SHAP summary plot."""
    st.subheader("What Features Matter Most for All Predictions?")
    
    # Use a subset of data for performance
    shap_data = data.sample(n=100, random_state=42) if len(data) > 100 else data
    
    explainer = shap.Explainer(model, shap_data)
    shap_values = explainer(shap_data)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, shap_data, plot_type="bar", show=False)
    st.pyplot(fig)

    st.info("""
        **How to read this chart:**
        - This chart shows the average impact of each feature on the model's prediction (magnitude, not direction).
        - Higher values mean the feature has a greater influence on the price prediction overall.
    """)

def parse_total_sqft(value):
    """Parses total_sqft values, handling ranges, square meters, and other formats."""
    try:
        value = str(value).strip()
        # Handle ranges (e.g., "1000-2000")
        if '-' in value:
            low, high = value.split('-')
            return (float(low.strip()) + float(high.strip())) / 2
        # Handle square meters (e.g., "34.46Sq. Meter")
        if 'sq. meter' in value.lower():
            num = float(re.findall(r'[\d.]+', value)[0])
            return num * 10.7639  # Convert to square feet
        # Handle other units or invalid formats
        return float(re.findall(r'[\d.]+', value)[0])
    except (ValueError, IndexError):
        return None  # Return None for invalid values

def clean_data_for_comparison(data):
    """Preprocesses data for comparison plots."""
    df_comp = data.copy()
    df_comp.dropna(inplace=True)
    # Convert total_sqft to numeric
    df_comp['total_sqft'] = df_comp['total_sqft'].apply(parse_total_sqft)
    # Drop rows where total_sqft couldn't be parsed
    df_comp = df_comp.dropna(subset=['total_sqft'])
    df_comp['price_per_sqft'] = df_comp['price'] * 100000 / df_comp['total_sqft']
    # Extract BHK
    df_comp['bhk'] = df_comp['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else None)
    df_comp = df_comp.dropna(subset=['bhk'])
    return df_comp

def display_comparison_plots(df):
    """Displays plots to compare different locations."""
    st.header("Compare House Features Across Locations")
    
    df_comp = clean_data_for_comparison(df)
    
    locations = sorted(df_comp['location'].unique())
    
    # User selections
    selected_locations = st.multiselect("Select locations to compare", locations, default=["Whitefield", "Sarjapur  Road"])
    
    feature_to_compare = st.selectbox(
        "Select a feature to compare",
        ['price_per_sqft', 'price', 'bhk', 'bath']
    )
    
    if selected_locations:
        filtered_df = df_comp[df_comp['location'].isin(selected_locations)]
        
        st.subheader(f"{feature_to_compare.replace('_', ' ').title()} Comparison")
        
        # Boxplot for distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=filtered_df, x='location', y=feature_to_compare, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Barplot for average values
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=filtered_df, x='location', y=feature_to_compare, ax=ax, estimator=np.mean)
        plt.title(f"Average {feature_to_compare.replace('_', ' ').title()} by Location")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one location to compare.")

def haversine(coord1, coord2):
    """Calculates the Haversine distance between two coordinates."""
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_location_stats(data, coords):
    """Calculates location statistics for map visualization."""
    df_stats = data.copy()
    df_stats.dropna(subset=['location', 'total_sqft', 'price'], inplace=True)
    df_stats['total_sqft'] = df_stats['total_sqft'].apply(parse_total_sqft)
    # Drop rows where total_sqft couldn't be parsed
    df_stats = df_stats.dropna(subset=['total_sqft'])
    df_stats['price_per_sqft'] = df_stats['price'] * 100000 / df_stats['total_sqft']
    
    location_stats = df_stats.groupby('location')['price_per_sqft'].mean().reset_index()
    
    coord_df = pd.DataFrame(coords.items(), columns=['location', 'coordinates'])
    coord_df['lat'] = coord_df['coordinates'].apply(lambda x: x[0])
    coord_df['lon'] = coord_df['coordinates'].apply(lambda x: x[1])
    
    map_data = pd.merge(location_stats, coord_df, on='location')
    return map_data

def display_recommendations(user_location, location_stats_df):
    """Displays cheaper nearby locations."""
    st.header("✨ Cheaper Nearby Locations")
    
    if user_location not in location_stats_df['location'].values:
        st.warning(f"Sorry, we don't have enough data for '{user_location}' to provide recommendations.")
        return

    user_loc_data = location_stats_df[location_stats_df['location'] == user_location].iloc[0]
    
    # Calculate distances
    distances = location_stats_df.apply(
        lambda row: haversine((user_loc_data['lat'], user_loc_data['lon']), (row['lat'], row['lon'])),
        axis=1
    )
    
    # Find cheaper locations
    cheaper_locations = location_stats_df[location_stats_df['price_per_sqft'] < user_loc_data['price_per_sqft']].copy()
    cheaper_locations['distance_km'] = distances
    
    # Sort by distance and get top 3
    recommendations = cheaper_locations.sort_values(by='distance_km').head(4) # 1 is self
    recommendations = recommendations[recommendations['location'] != user_location].head(3)

    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            st.subheader(f"{row['location']} ({row['distance_km']:.2f} km away)")
            st.metric(
                label="Average Price per Sqft",
                value=f"₹{row['price_per_sqft']:,.2f}",
                delta=f"₹{row['price_per_sqft'] - user_loc_data['price_per_sqft']:,.2f} (cheaper)"
            )
    else:
        st.info("Your selected location is already one of the cheapest in its vicinity!")

def display_map_visualization(map_df):
    """Displays a map visualization of house prices."""
    st.header("Map Visualization of Property Prices")

    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=10,
        pitch=50,
    )

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=map_df,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius='price_per_sqft / 50',
        pickable=True
    )

    tooltip = {
        "html": "<b>Location:</b> {location} <br/> <b>Avg. Price/Sqft:</b> ₹{price_per_sqft:.2f}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/light-v9'
    )

    st.pydeck_chart(r)
    st.info("The size of the circle corresponds to the average price per square foot in that location.")

def generate_report_content(user_inputs, prediction, shap_values):
    """Generates a string for the downloadable report."""
    report = "Bengaluru House Price Prediction Report\n"
    report += "="*40 + "\n\n"
    
    report += "Input Features:\n"
    for key, value in user_inputs.items():
        report += f"- {key.replace('_', ' ').title()}: {value}\n"
    report += "\n"
    
    report += f"Predicted Price: ₹ {prediction:,.2f}\n"
    report += "="*40 + "\n\n"
    
    report += "Prediction Explanation (Top Influencers):\n"
    
    # Process SHAP values
    shap_df = pd.DataFrame({
        'feature': shap_values.feature_names,
        'value': shap_values.values[0],
    })
    
    positive_impact = shap_df[shap_df['value'] > 0].sort_values(by='value', ascending=False).head(3)
    negative_impact = shap_df[shap_df['value'] < 0].sort_values(by='value', ascending=True).head(3)
    
    report += "\nFeatures Increasing the Price:\n"
    if not positive_impact.empty:
        for _, row in positive_impact.iterrows():
            report += f"- {row['feature']}\n"
    else:
        report += "- None\n"
        
    report += "\nFeatures Decreasing the Price:\n"
    if not negative_impact.empty:
        for _, row in negative_impact.iterrows():
            report += f"- {row['feature']}\n"
    else:
        report += "- None\n"
        
    return report

if __name__ == "__main__":
    main()