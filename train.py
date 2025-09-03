import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Debugging: Check initial data
print("Initial data shape:", df.shape)
print("Initial columns:", df.columns.tolist())

# 1️⃣ Preprocessing
# Drop irrelevant columns
df = df.drop(['area_type', 'society'], axis=1, errors='ignore')

# Handle missing values
df = df.dropna(subset=['location', 'bath', 'balcony', 'total_sqft', 'price'])
print("Shape after dropping missing values:", df.shape)

# Convert 'total_sqft' to numeric, handle ranges
def convert_sqft(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])
print("Shape after total_sqft cleaning:", df.shape)

# Create 'bhk' from total_sqft
df['bhk'] = (df['total_sqft'] / 300).apply(np.floor)

# Remove outliers
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
df = df[(df['price_per_sqft'] > 1000) & (df['price_per_sqft'] < 20000)]
df = df[df['total_sqft'] / df['bhk'] >= 300]
print("Shape after outlier removal:", df.shape)

# Clean location names
df['location'] = df['location'].apply(lambda x: x.strip())
location_counts = df['location'].value_counts()
rare_locations = location_counts[location_counts <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)
print("Unique locations:", df['location'].nunique())

# One-hot encode locations with integer dtype
dummies = pd.get_dummies(df['location'], dtype=int)
df = pd.concat([df, dummies.drop('other', axis=1)], axis=1)
print("Shape after one-hot encoding:", df.shape)
print("One-hot encoded columns:", dummies.columns.tolist())

# Proximity Feature: Distance to MG Road
mg_road_coords = (12.9756, 77.6047)
location_coords = {
    "Whitefield": (12.9698, 77.7499),
    "Sarjapur  Road": (12.9121, 77.6844),
    "Electronic City": (12.8412, 77.6636),
    "Marathahalli": (12.9592, 77.6974),
    "Hebbal": (13.0350, 77.5970),
    "HSR Layout": (12.9121, 77.6446),
    "Indira Nagar": (12.9784, 77.6408),
    "Yelahanka": (13.1007, 77.5963),
    "Koramangala": (12.9352, 77.6245),
    "Jayanagar": (12.9293, 77.5822),
    "Rajaji Nagar": (12.9901, 77.5529),
    "Bellandur": (12.9304, 77.6784),
    "Bannerghatta Road": (12.8884, 77.6039),
    "1st Block Jayanagar": (12.9293, 77.5822)
}

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_distance_to_center(loc):
    if loc in location_coords:
        return haversine(location_coords[loc], mg_road_coords)
    else:
        return np.nan

df["distance_to_center_km"] = df["location"].apply(get_distance_to_center)

# Neighborhood Feature: Average price per sqft
df["location_avg_price_per_sqft"] = df.groupby("location")["price_per_sqft"].transform('mean')

# Temporal Feature: Extract and impute year
def extract_year(availability):
    if isinstance(availability, str):
        if availability.isdigit():
            return int(availability)
        elif '-' in availability:
            try:
                year_suffix = availability.split('-')[1]
                if len(year_suffix) == 3:  # e.g., 'May'
                    year_prefix = availability.split('-')[0]
                    if year_prefix.isdigit():
                        year = int(year_prefix)
                        return 2000 + year if year < 100 else year
            except:
                return np.nan
        elif availability == "Ready To Move":
            return 2020
    return np.nan

if "availability" in df.columns:
    df["year"] = df["availability"].apply(extract_year)
    df["year"] = df["year"].fillna(df["year"].median() if not df["year"].isna().all() else 2020)
    print("Year missing values:", df["year"].isna().sum())
    print("Year sample values:", df["year"].head().tolist())

# Amenity distances
amenity_distances = {
    "Whitefield": {"school": 1.2, "hospital": 2.5, "metro": 3.0},
    "Sarjapur Road": {"school": 0.8, "hospital": 1.5, "metro": 2.2},
    "Electronic City": {"school": 1.0, "hospital": 2.0, "metro": 3.5},
    "Hebbal": {"school": 1.5, "hospital": 2.0, "metro": 2.0},
    "Yelahanka": {"school": 1.2, "hospital": 1.8, "metro": 2.5},
    "Marathahalli": {"school": 1.0, "hospital": 1.8, "metro": 2.8},
    "Indira Nagar": {"school": 0.5, "hospital": 1.0, "metro": 1.5},
    "HSR Layout": {"school": 0.8, "hospital": 1.2, "metro": 2.0},
    "1st Block Jayanagar": {"school": 0.7, "hospital": 1.0, "metro": 1.2},  # Added
    "other": {"school": 2.0, "hospital": 2.5, "metro": 3.0}
}

def get_school_distance(loc):
    return amenity_distances.get(loc, amenity_distances["other"])["school"]

def get_hospital_distance(loc):
    return amenity_distances.get(loc, amenity_distances["other"])["hospital"]

def get_metro_distance(loc):
    return amenity_distances.get(loc, amenity_distances["other"])["metro"]

df["school_dist_km"] = df["location"].apply(get_school_distance)
df["hospital_dist_km"] = df["location"].apply(get_hospital_distance)
df["metro_dist_km"] = df["location"].apply(get_metro_distance)

# Log-transform price (assuming price is in lakhs)
df['price'] = np.log(df['price'])
print("Price (log-transformed) sample:", df['price'].head().tolist())

# Prepare features
X = df.drop(['price', 'price_per_sqft', 'location', 'availability', 'size'], axis=1, errors='ignore')
y = df['price']
print("X columns:", X.columns.tolist())

# Ensure 'year' is in X
if "year" not in X.columns:
    X["year"] = df["year"]

# Identify numeric features
numeric_features = ['total_sqft', 'bath', 'balcony', 'bhk',
                    'distance_to_center_km', 'location_avg_price_per_sqft',
                    'year', 'school_dist_km', 'hospital_dist_km', 'metro_dist_km']

existing_numeric_features = [col for col in numeric_features if col in X.columns]
categorical_features = [col for col in X.columns if col not in existing_numeric_features]
print("Numeric features:", existing_numeric_features)
print("Categorical features:", categorical_features)

# Impute numeric values
numeric_imputer = SimpleImputer(strategy='median')
X[existing_numeric_features] = pd.DataFrame(
    numeric_imputer.fit_transform(X[existing_numeric_features]),
    columns=existing_numeric_features,
    index=X.index
)

# Impute categorical values with 0
if categorical_features:
    X[categorical_features] = X[categorical_features].astype(int)
    categorical_imputer = SimpleImputer(strategy='constant', fill_value=0)
    X[categorical_features] = pd.DataFrame(
        categorical_imputer.fit_transform(X[categorical_features]),
        columns=categorical_features,
        index=X.index
    )

# Scale numeric features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[existing_numeric_features] = scaler.fit_transform(X[existing_numeric_features])
print("Scaler feature names:", getattr(scaler, 'feature_names_in_', existing_numeric_features))

# Split data
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train_tree, X_test_tree, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models_to_tune = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42)
}

param_grids = {
    "Random Forest": {
        'n_estimators': [100, 150],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "XGBoost": {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 6]
    }
}

# --- Model Training and Tuning ---
final_models = {}

# Train and tune tree-based models on unscaled data
for name, model in models_to_tune.items():
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],
        n_iter=4,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_tree, y_train)
    final_models[name] = random_search.best_estimator_
    print(f"Best parameters for {name}: {random_search.best_params_}")

# Train Linear Regression on scaled data
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
final_models["Linear Regression"] = lr

# --- Model Evaluation ---
results = {}
for name, model in final_models.items():
    if name == "Linear Regression":
        X_te = X_test_scaled
    else:
        X_te = X_test_tree
        
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# Select best model based on R²
best_model_name = max(results, key=lambda x: results[x]['R²'])
best_model = final_models[best_model_name]
print(f"Best Model: {best_model_name}")

# Save artifacts
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
joblib.dump(existing_numeric_features, "numeric_features.pkl")

# Save test data for SHAP explanations in the app
joblib.dump(X_test_tree, "X_test_tree.pkl")
joblib.dump(location_coords, "location_coords.pkl")