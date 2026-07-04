from pathlib import Path
import joblib
import pandas as pd
import sys

BASE_DIR = Path.cwd()
PROJECT_ROOT = Path(r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml")
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgboost.joblib"
INPUT_PATH = PROJECT_ROOT / "data" / "input" / "new_properties.json"

def predict_new_data(json_input_path):
    """
    Loads a pre-trained machine learning pipeline, reads property data from a JSON file,
    generates price predictions, and outputs the results to the console.
    Args: json_input_path: the file path to the JSON containing property data.
    Returns: pd.DataFrame: a df containing the original data with an appended 'predicted_price' column.
    """
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    # Load pipeline
    pipeline = joblib.load(MODEL_PATH)
    
    # Load input data
    try:
        new_data = pd.read_json(json_input_path)
    except ValueError as e:
        print(f"Error reading JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Predict: TransformedTargetRegressor automatically handles inverse transformation
    predictions = pipeline.predict(new_data)
    
    new_data['predicted_price'] = predictions
    
    for i, price in enumerate(predictions):
        print(f"Estimation for property {i+1}: €{price:,.2f}")
    
    return new_data

if __name__ == "__main__":
    if INPUT_PATH.exists():
        predict_new_data(INPUT_PATH)
    else:
        print(f"Error: Input file not found at {INPUT_PATH}")
