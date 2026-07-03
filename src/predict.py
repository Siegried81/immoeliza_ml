import joblib
import pandas as pd
import numpy as np

MODEL_PATH = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models\best_xgboost.joblib"
INPUT_PATH = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\input\new_properties.json"

def predict_new_data(json_input_path):
    """
    Loads a pre-trained machine learning pipeline and generates price predictions 
    for new property data provided in a JSON file format.
    """
    pipeline = joblib.load(MODEL_PATH)
    new_data = pd.read_json(json_input_path)
    log_predictions = pipeline.predict(new_data)
    
    # Revert from log scale
    predictions = np.expm1(log_predictions)
    
    new_data['predicted_price'] = predictions
    print(f"Estimation for the property: €{predictions[0]:,.2f}")

if __name__ == "__main__":
    predict_new_data(INPUT_PATH)