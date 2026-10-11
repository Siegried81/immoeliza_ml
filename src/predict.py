import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# 1. Path Definitions
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_INPUT = BASE_DIR / "data" / "input"
DATA_OUTPUT = BASE_DIR / "data" / "output"
MODEL_PATH = MODEL_DIR / "best_xgboost.joblib"
INPUT_PATH = DATA_INPUT / "new_properties.json"

def run_prediction():
    # 2. Load the model
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    pipeline = joblib.load(MODEL_PATH)

    # 3. Load the data
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found at {INPUT_PATH}")
        return
        
    df_new = pd.read_json(INPUT_PATH)

    # 4. Prepare features (Drop ID if present)
    # We create the 'features' variable here so it is defined before .predict()
    features = df_new.drop(columns=['property_id'], errors='ignore')

    # 5. Predict
    # We use pipeline.predict directly. 
    # If your prices are still around 12-13, use np.expm1(predictions)
    predictions = pipeline.predict(features)
    
    # If the results are ~12, use this:
    if predictions.mean() < 20:
        print("Note: Results appear to be in log scale. Applying inverse transformation...")
        predictions = np.expm1(predictions)

    # 6. Save and Print
    df_new['predicted_price'] = predictions
    
    # Ensure output directory exists
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    df_new.to_json(DATA_OUTPUT / "predictions.json", orient="records", indent=4)
    
    for _, row in df_new.iterrows():
        print(f"Prediction for {row['property_id']}: {row['predicted_price']:,.2f} €")

if __name__ == "__main__":
    run_prediction()