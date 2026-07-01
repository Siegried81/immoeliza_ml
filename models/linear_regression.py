import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"

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

    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Perfect prediction")

    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def run_simple_regression():
    """Train and evaluate a simple linear regression model."""
    
    # Ensure model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Load and prepare data
    if not os.path.exists(OUTPUT_FILE):
        print(f"Error: File not found at {OUTPUT_FILE}")
        return

    df = pd.read_json(OUTPUT_FILE)
    df = df.dropna(subset=["price", "livable_surface"])

    # Define features and target
    X = df[["livable_surface"]]
    y = df["price"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "linear_regression_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to:\n{model_path}")

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    display_metrics(y_test, y_pred, "Linear Regression")
    plot_actual_vs_predicted(y_test, y_pred, "Linear Regression")

if __name__ == "__main__":
    run_simple_regression()