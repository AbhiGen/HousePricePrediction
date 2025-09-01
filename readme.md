# Bengaluru House Price Prediction

## Overview
This project is an end-to-end machine learning pipeline for predicting house prices in Bengaluru, India, using the Bengaluru House Price Dataset. The goal is to build a robust regression model to predict the **logarithm of house prices (price)** based on features like total square footage, number of bathrooms, balconies, bedrooms (BHK), and location-based features.  

The pipeline includes:

- **Data Preprocessing:** Handling missing values, outliers, and inconsistent data formats.  
- **Feature Engineering:** Creating domain-specific features like distance to city center and proximity to amenities.  
- **Model Training:** Training and tuning Linear Regression, Random Forest, and XGBoost (if available) models.  
- **Model Evaluation:** Assessing performance using RMSE, MAE, and R² metrics.  
- **Visualizations:** Generating plots for predicted vs. actual prices, residuals, feature importance, and model comparison.  

The project is designed for scalability, with potential for deployment as a web application (e.g., using Flask, Docker, or Heroku).

---

## Dataset

**Source:** Bengaluru House Price Dataset on Kaggle  
**Shape:** 13,320 rows, 9 columns (initially)  

**Features:**
- `area_type`: Type of area (e.g., Super built-up Area)  
- `availability`: Availability status (e.g., Ready To Move, 19-Dec)  
- `location`: Property location in Bengaluru  
- `size`: BHK information (e.g., "2 BHK")  
- `society`: Name of the housing society  
- `total_sqft`: Total area in square feet (string, includes ranges)  
- `bath`: Number of bathrooms  
- `balcony`: Number of balconies  
- `price`: Target variable (price in lakhs)  

**Missing Values:**
- `society`: 5,502 missing  
- `balcony`: 609 missing  
- `bath`: 73 missing  
- `size`: 16 missing  
- `location`: 1 missing  

---

## Project Structure

```

```


---

## Preprocessing Steps

1. **Drop Irrelevant Columns**
   - Dropped: `area_type`, `society`
   - Reason: Low predictive power or too many missing values  

2. **Handle Missing Values**
   - Dropped rows with nulls in essential columns (`location`, `bath`, `balcony`, `total_sqft`, `price`)  

3. **Convert `total_sqft` to Numeric**
   - Convert ranges (e.g., "1200-1500" → 1350)  
   - Drop invalid entries  

4. **Create `bhk` Feature**
   - Derive `bhk` by dividing `total_sqft` by 300 and rounding down  

5. **Outlier Removal**
   - Remove rows where `price_per_sqft` < 1,000 or > 20,000 INR  
   - Remove rows where `total_sqft` / `bhk` < 300  

6. **Clean `location` Column**
   - Strip whitespace  
   - Group locations with ≤ 10 occurrences into "other"  

7. **One-Hot Encode `location`**
   - Drop "other" to avoid multicollinearity  

8. **Log-Transform `price`**
   - Apply `np.log` to reduce skewness  

9. **Feature Engineering**
   - `distance_to_center_km`: Distance to MG Road (city center)  
   - `location_avg_price_per_sqft`: Mean price per sqft per location  
   - `year`: Extracted from availability (mostly NaN)  
   - `school_dist_km`, `hospital_dist_km`, `metro_dist_km`: Distances to amenities  

10. **Imputation and Scaling**
    - Impute numeric features with median, categorical features with 0  
    - Scale numeric features for Linear Regression  

✅ **Final Dataset:** 12,434 rows, 243 features  
- Features: `total_sqft`, `bath`, `balcony`, `bhk`, `distance_to_center_km`, `location_avg_price_per_sqft`, `year`, `school_dist_km`, `hospital_dist_km`, `metro_dist_km`, one-hot encoded locations  
- Target: `log(price)`

---

## Model Training

**Models Trained:**
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor (optional)

**Hyperparameter Tuning:**
- RandomizedSearchCV for Random Forest and XGBoost  
- Linear Regression: default parameters  

**Evaluation Metrics:**
- RMSE: Root Mean Squared Error  
- MAE: Mean Absolute Error  
- R²: Coefficient of determination  

---

## Visualizations

- **Correlation Heatmap:** Highlights strong predictors like `price_per_sqft`, `bhk`, `total_sqft`  
- **Predicted vs Actual Scatter Plot:** Compares predictions with true log(price)  
- **Residual Plot:** Shows errors to detect model bias  
- **Feature Importance:** Top 10 features for Random Forest and XGBoost  
- **Model Comparison:** R², RMSE, and MAE across models  

---

## Installation

**Clone the Repository:**
```bash
git clone https://github.com/your-username/Bengaluru_House_Price_Project.git
cd Bengaluru_House_Price_Project

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

pip install -r requirements.txt
