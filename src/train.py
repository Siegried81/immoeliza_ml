import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from src.preprocessing import get_preprocessor 
import sys
from pathlib import Path
# The repository root on sys.path, so `src.paths` resolves whether this file is
# run through main.py or on its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

OUTPUT_FILE = paths.CLEAN_DATA
MODEL_DIR = paths.MODEL_DIR
MODEL_PATH = paths.MODEL_PATH

def train_xgboost():
    """
    Trains a full pipeline containing the preprocessor and the XGBoost regressor.
    Evaluates model performance using MAE, RMSE, and R-squared metrics.
    """
    df = pd.read_json(OUTPUT_FILE)
    
    num_cols = ['postcode', 'latitude', 'longitude', 'build_year', 'bedroom_count', 
                'livable_surface', 'total_surface', 'garage', 'terrace', 'swimming_pool', 
                'energy_consumption_kWh/m2/year', 'preschool_distance_m', 'train_station_distance_m', 
                'supermarket_distance_m', 'nearest_city_distance_km']
    cat_cols = ['property_type', 'city', 'province', 'property_state', 'nearest_city']
    
    X = df[num_cols + cat_cols]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor(num_cols, cat_cols)),
        ('regressor', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
    ])
    
    # Train the pipeline
    print("Training the pipeline")
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost()