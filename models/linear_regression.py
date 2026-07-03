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
IMAGE_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\images"

def display_metrics(y_test, y_pred, model_name="Model"):
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
    """Plot and save the comparison chart."""
    # Ensure images directory exists
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        
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
    
    # Save the plot
    save_path = os.path.join(IMAGE_DIR, f"{model_name.lower().replace(' ', '_')}_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    
    plt.show()

def run_simple_regression():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(OUTPUT_FILE):
        print(f"Error: File not found at {OUTPUT_FILE}")
        return

    df = pd.read_json(OUTPUT_FILE)
    df = df.dropna(subset=["price", "livable_surface"])

    X = df[["livable_surface"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Generate predictions (Explicitly define train_pred and y_pred)
    train_pred = model.predict(X_train)
    y_pred = model.predict(X_test) # Defining y_pred here fixes your error

    # Calculate R² scores for overfitting check
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"\nR² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")

    # Analysis logic
    if abs(r2_train - r2_test) < 0.05:
        print("The model is well-balanced (no significant overfitting).")
    else:
        print("Warning: The model shows signs of overfitting.")

    # Save model
    model_path = os.path.join(MODEL_DIR, "linear_regression_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to:\n{model_path}")

    # Evaluate and plot
    display_metrics(y_test, y_pred, "Linear Regression")
    plot_actual_vs_predicted(y_test, y_pred, "Linear Regression")

if __name__ == "__main__":
    run_simple_regression()
