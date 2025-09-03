```
Bengaluru House Price Predictor
Overview
The Bengaluru House Price Predictor is an interactive web application built using Streamlit that predicts house prices in Bengaluru, India, based on user-provided features such as total square footage, number of bedrooms (BHK), bathrooms, balconies, location, and additional attributes like proximity to amenities. The application leverages a machine learning model trained on the Bengaluru House Prices dataset, enhanced with feature engineering, and provides insights through SHAP explanations, location-based comparisons, and a map visualization of average property prices.
Key features include:

Price Prediction: Predicts house prices using a trained machine learning model (e.g., Random Forest, XGBoost, or Linear Regression).
Feature Importance: Uses SHAP to explain individual predictions and global feature importance.
Location Comparison: Visualizes price and feature distributions across different locations.
Map Visualization: Displays a map with average price per square foot by location using PyDeck.
Recommendations: Suggests cheaper nearby locations based on user-selected location.
Downloadable Reports: Generates a downloadable text report summarizing predictions and key influencing features.

Prerequisites
Ensure you have the following installed:

Python 3.8 or higher
pip (Python package manager)

Installation

Clone the Repository (if hosted on a version control platform like GitHub):
git clone https://github.com/your-username/bengaluru-house-price-predictor.git
cd bengaluru-house-price-predictor


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Create a requirements.txt file with the following content:
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
shap==0.46.0
matplotlib==3.9.2
seaborn==0.13.2
pydeck==0.9.1
scikit-learn==1.5.1
xgboost==2.1.1

Then, install the dependencies:
pip install -r requirements.txt


Download the Dataset:

The project uses the Bengaluru_House_Data.csv dataset, which should be placed in the project root directory.
You can obtain the dataset from Kaggle or another reliable source.


Prepare Artifacts:

Run the train_model.py script to generate the required model artifacts (best_model.pkl, scaler.pkl, feature_columns.pkl, numeric_features.pkl, X_test_tree.pkl, location_coords.pkl).
Alternatively, ensure these artifacts are already present in the project directory.



Project Structure
bengaluru-house-price-predictor/
├── app.py                    # Main Streamlit application
├── train_model.py            # Script to preprocess data, train models, and save artifacts
├── Bengaluru_House_Data.csv  # Dataset for training and analysis
├── best_model.pkl            # Trained machine learning model
├── scaler.pkl                # StandardScaler for numeric feature scaling
├── feature_columns.pkl       # List of feature columns used in the model
├── numeric_features.pkl      # List of numeric features for scaling
├── X_test_tree.pkl           # Test data for SHAP explanations
├── location_coords.pkl       # Dictionary of location coordinates
├── requirements.txt          # Python dependencies
└── README.md                 # This file

Usage

Run the Training Script:

Execute train_model.py to preprocess the dataset, train the machine learning models, and generate the necessary artifacts:python train_model.py


This script performs data cleaning, feature engineering (e.g., handling total_sqft formats, adding proximity features), trains multiple models (Random Forest, XGBoost, Linear Regression), selects the best model based on R² score, and saves the artifacts.


Run the Streamlit App:

Start the Streamlit application:streamlit run app.py


Open your web browser and navigate to http://localhost:8501 to access the app.


Interact with the App:

Price Predictor Tab:
Use the sidebar to input house features (e.g., total square footage, BHK, location).
Click "Predict Price" to get the predicted house price, SHAP waterfall plot, and a downloadable report.
Click "Get Recommendations" to view cheaper nearby locations.
Expand "View Global Feature Importance" to see which features most influence predictions.


House Comparison Tab:
Select locations and a feature (e.g., price per sqft, BHK) to compare distributions via boxplots and barplots.


Map Visualization Tab:
View a map of Bengaluru with circles representing average price per square foot by location.





Data Preprocessing and Feature Engineering
The train_model.py script performs the following steps:

Data Cleaning:
Drops irrelevant columns (area_type, society).
Removes rows with missing critical values (location, bath, balcony, total_sqft, price).
Converts total_sqft to numeric, handling ranges (e.g., "1000-2000") and invalid formats.
Removes outliers based on price_per_sqft and total_sqft per BHK.
Groups rare locations (fewer than 10 occurrences) into an "other" category.


Feature Engineering:
Creates bhk from total_sqft (assuming 300 sqft per BHK).
Adds distance_to_center_km using Haversine distance to MG Road.
Computes location_avg_price_per_sqft as the mean price per square foot per location.
Extracts year from availability, with imputation for missing values.
Adds amenity distances (school_dist_km, hospital_dist_km, metro_dist_km) based on predefined values.


Feature Encoding:
One-hot encodes location names.
Scales numeric features using StandardScaler.


Model Training:
Trains Random Forest, XGBoost, and Linear Regression models.
Uses RandomizedSearchCV to tune hyperparameters for tree-based models.
Selects the best model based on R² score.



Notes

Dataset: The Bengaluru_House_Data.csv dataset must be in the project root directory. Ensure it includes columns like location, total_sqft, bath, balcony, price, size, and availability.
Artifacts: The model artifacts must match the feature set used in app.py. Re-run train_model.py if you modify the dataset or preprocessing steps.
Map Visualization: The map uses PyDeck and requires valid coordinates in location_coords.pkl. The default coordinates are centered on Bengaluru (latitude: 12.9716, longitude: 77.5946).
SHAP Explanations: SHAP plots may take time to render for large datasets. The app uses a sampled subset of test data for global feature importance to improve performance.
Error Handling: The app includes robust parsing for total_sqft to handle formats like ranges and square meters (converted to square feet using 1 sq. meter = 10.7639 sq. feet).

Troubleshooting

FileNotFoundError: Ensure all required files (Bengaluru_House_Data.csv, best_model.pkl, etc.) are in the project directory.
ValueError in total_sqft: The parse_total_sqft function handles various formats, but ensure the dataset does not contain unexpected formats. Check train_model.py output for debugging.
Streamlit Errors: Verify that all dependencies are installed correctly. Use pip install -r requirements.txt to ensure compatibility.
Map Not Rendering: Ensure pydeck is installed and location_coords.pkl contains valid coordinates.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a clear description of changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset sourced from Kaggle.
Built with Streamlit, SHAP, and PyDeck.
Inspired by real-world applications of machine learning in real estate.

```