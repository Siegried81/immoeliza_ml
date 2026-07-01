import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_PATH = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")

def display_metrics(y_test, y_pred, model_name="Model"):
    """Display regression performance metrics."""
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Metrics")
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:,.2f}")
    print(f"RMSE     : €{rmse:,.2f}")
    print(f"MAE      : €{mae:,.2f}")

def plot_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    """Plot actual prices versus predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="blue", edgecolors="black")
    
    line_min, line_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Perfect prediction")
    
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def train_and_evaluate_rf():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    df = pd.read_json(DATA_PATH)
    target = 'price'
    df = df[df[target] > 0].copy()
    
    X = df.drop(columns=[target, 'address'], errors='ignore')
    y_log = np.log(df[target])
    
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_log)
    
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    y_pred_log = rf_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_test = np.exp(y_test_log)
    
    display_metrics(y_test, y_pred, "Random Forest")
    plot_actual_vs_predicted(y_test, y_pred, "Random Forest")

if __name__ == "__main__":
    train_and_evaluate_rf()