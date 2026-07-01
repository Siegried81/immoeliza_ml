import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

# Configuration
OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"

def display_metrics(y_test, y_pred, model_name="Model"):
    """Display regression performance metrics."""
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{model_name} Metrics")
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:,.2f}")
    print(f"RMSE     : €{rmse:,.2f}")
    print(f"MAE      : €{mae:,.2f}")

def plot_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    """Plot actual prices versus predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="darkblue", edgecolors="black")
    line_min, line_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def run_xgboost_pipeline():
    """Train, save, and evaluate the optimized XGBoost pipeline."""
    
    # Ensure directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load and Clean
    df = pd.read_json(OUTPUT_FILE).dropna(subset=['price'])
    df = df[df['price'] > 0]
    X = df.drop(columns=['price', 'address'], errors='ignore')
    y = df['price']

    # 2. Setup Preprocessing
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough')

    # 3. Build Pipeline with TransformedTargetRegressor (Log transformation)
    # Cela permet d'entraîner sur log1p(y) et de prédire sur expm1(y) automatiquement
    model = Pipeline([
        ('prep', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(
                n_estimators=1000, 
                learning_rate=0.05, 
                max_depth=6, 
                random_state=42
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    # 4. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # 5. Save
    model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    joblib.dump(model, model_path)
    print(f"XGBoost model saved to: {model_path}")

    # 6. Evaluate
    y_pred = model.predict(X_test)
    display_metrics(y_test, y_pred, "XGBoost Regression")
    plot_actual_vs_predicted(y_test, y_pred, "XGBoost Regression")

if __name__ == "__main__":
    run_xgboost_pipeline()