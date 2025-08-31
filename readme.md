# Bengaluru House Price Prediction

## Overview
This project is an **end-to-end machine learning pipeline** for predicting house prices in Bengaluru, India.  
It includes:

- Data preprocessing (handling missing values, outliers, feature engineering)
- Regression model training
- Saving the trained model using pickle
- Flask API for serving predictions
- Ready for Docker and Heroku deployment

---

## Dataset
- Source: [Bengaluru House Price Dataset on Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)  
- Features include:
  - `location` – property location  
  - `total_sqft` – total area in sqft  
  - `bath` – number of bathrooms  
  - `balcony` – number of balconies  
  - `size` – BHK information  
  - `price` – target variable (price in lakhs)  

---

## Project Structure

"""
Bengaluru_House_Price_Project/
│── data/ # Dataset CSV
│── src/ # Python scripts
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── utils.py
│── app.py # Flask API
│── requirements.txt
│── Dockerfile
│── Procfile # For Heroku deployment
│── model.pkl # Pickled trained model
│── hp.ipynb # Jupyter Notebook for exploration
└── README.md
"""


---

## Preprocessing Steps

Raw Bengaluru Housing Dataset (13320 rows, 9 columns)
                 │
                 ▼
1️⃣ Drop Irrelevant Columns
   - area_type, availability, society, size
                 │
                 ▼
2️⃣ Handle Missing Values
   - Drop rows with nulls in location, bath, balcony, total_sqft
                 │
                 ▼
3️⃣ Convert total_sqft to Numeric
   - Convert ranges like "1200-1500" → average
   - Remove non-numeric entries
                 │
                 ▼
4️⃣ Create bhk Feature
   - Extract number of bedrooms from size column (e.g., "2 BHK" → 2)
                 │
                 ▼
5️⃣ Outlier Removal
   - Remove properties with price_per_sqft < 1000 or > 20000
   - Remove rows where total_sqft / bhk < 300
                 │
                 ▼
6️⃣ Clean Location Column
   - Strip extra spaces
   - Group rare locations (≤ 10 occurrences) into 'other'
                 │
                 ▼
7️⃣ One-Hot Encode Locations
   - Convert categorical location column into numeric columns
   - Drop 'other' column to avoid multicollinearity
                 │
                 ▼
8️⃣ Optional Target Transformation
   - Apply log transformation to price to reduce skewness
                 │
                 ▼
✅ Final Preprocessed Dataset (12434 rows, 235 features)
   - Features: total_sqft, bath, balcony, bhk, location dummies
   - Target: log(price)
