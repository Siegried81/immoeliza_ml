import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from pathlib import Path
from preprocessing import get_preprocessor 

# Path Definitions
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_CLEAN = BASE_DIR / "data" / "clean"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_xgboost.joblib"
OUTPUT_FILE = DATA_CLEAN / "cleaned_data.json"

def train_xgboost():
    """
    Trains a full pipeline containing the preprocessor and the XGBoost regressor.
    Targets are log-transformed (log1p) to better handle price distribution.
    """
    if not OUTPUT_FILE.exists():
        print(f"Error: Cleaned data not found at {OUTPUT_FILE}")
        return

    df = pd.read_json(OUTPUT_FILE)
    
    num_cols = ['postcode', 'latitude', 'longitude', 'build_year', 'bedroom_count', 
                'livable_surface', 'total_surface', 'garage', 'terrace', 'swimming_pool', 
                'energy_consumption_kWh/m2/year', 'preschool_distance_m', 'train_station_distance_m', 
                'supermarket_distance_m', 'nearest_city_distance_km']
    cat_cols = ['property_type', 'city', 'province', 'property_state', 'nearest_city']
    
    X = df[num_cols + cat_cols]
    # Transform price to log scale to compress the range and improve luxury prediction
    y = np.log1p(df['price'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor(num_cols, cat_cols)),
        ('regressor', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
    ])
    
    # Train the pipeline
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate (results will be in log scale)
    y_pred_log = pipeline.predict(X_test)
    
    # Convert back to real prices for evaluation
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    
    # Every headline metric is computed on the real euro scale. R² on the log
    # scale is a different, more flattering number: log1p compresses the
    # expensive tail, so the variance left to explain is much smaller there.
    # Mixing the two made the model look better than its euro errors show.
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = root_mean_squared_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    # Kept as a secondary diagnostic only, because the model optimises log price.
    r2_log = r2_score(y_test, y_pred_log)
    # A percentage is what a reader compares between models, and it is the one
    # metric here that does not depend on the price level of the test sample.
    # The median is printed beside it because MAPE is pulled up by the cheap
    # end: a 30k error is 30% of a 100k flat and 3% of a 1M villa.
    ape = np.abs(y_test_real - y_pred_real) / y_test_real * 100

    print("\nModel Evaluation (Real Price Scale):")
    print(f"Held-out rows : {len(y_pred_real):,}")
    print(f"MAE  : {mae:,.2f} €")
    print(f"RMSE : {rmse:,.2f} €")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {ape.mean():.2f} %  (median {np.median(ape):.2f} %)")
    print(f"R² (log1p scale, diagnostic only) : {r2_log:.4f}")
    
    # Save the model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost()