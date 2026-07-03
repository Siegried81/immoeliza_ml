import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

# Paths definition
OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"


def display_metrics(y_true, y_pred, title):
    """Prints evaluation metrics including RMSE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred) # Added RMSE
    
    print(f"\n--- {title} Metrics ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: €{mae:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: €{rmse:,.2f}") 
    
def plot_actual_vs_predicted(y_true, y_pred, title):
    """Plots actual vs predicted prices."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{title}: Actual vs Predicted")
    plt.show()

def check_overfitting(pipeline, X_train, y_train, X_test, y_test):
    """Checks for overfitting by comparing training and testing R² scores."""
    train_score = r2_score(y_train, pipeline.predict(X_train))
    test_score = r2_score(y_test, pipeline.predict(X_test))
    
    print(f"\n--- Overfitting Diagnostic ---")
    print(f"Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
    
    if train_score > test_score + 0.1:
        print("Verdict: Overfitting detected (Gap > 10%)")
    else:
        print("Verdict: Model generalizes well.")

def run_xgboost_pipeline():
    """Train, save, and evaluate the optimized XGBoost pipeline."""
    # Load and filter data
    df = pd.read_json(OUTPUT_FILE).dropna(subset=['price'])
    df = df[df['price'] > 0]
    X = df.drop(columns=['price', 'address'], errors='ignore')
    y = df['price']

    # Setup Preprocessing
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough')

    # Build Pipeline with TransformedTargetRegressor (Log transformation)
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

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    joblib.dump(model, model_path)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    
    # Overfitting diagnostic
    check_overfitting(model, X_train, y_train, X_test, y_test)

    # Print Metrics and Plot
    display_metrics(y_test, y_pred, "XGBoost Regression")
    plot_actual_vs_predicted(y_test, y_pred, "XGBoost Regression")

if __name__ == "__main__":
    run_xgboost_pipeline()