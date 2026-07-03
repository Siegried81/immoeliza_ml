import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def display_metrics(y_test, y_pred, model_name="Model"):
    """
    Computes and displays regression performance metrics.
    Includes R², MSE, RMSE and MAE for evaluation.
    Used to compare model accuracy on test data.
    """
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Metrics")
    print("-" * 40)
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:,.2f}")
    print(f"RMSE     : €{rmse:,.2f}")
    print(f"MAE      : €{mae:,.2f}")


def plot_actual_vs_predicted(y_test, y_pred, model_name="Model", save_path=None):
    """
    Plots predicted values against actual values.
    Helps visualize model performance and error distribution.
    Optionally saves the figure as a PNG file.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        color="purple",
        edgecolors="black"
    )

    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())

    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        "r--",
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def run_linear_regression():
    """
    Trains a simple Linear Regression model using one feature.
    Splits data into train/test sets and evaluates performance.
    Saves model and generates performance visualization.
    """
    df = pd.read_json(INPUT_FILE)
    df = df.dropna(subset=["price", "livable_surface"])

    X = df[["livable_surface"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")

    if abs(r2_train - r2_test) < 0.1:
        print("Model is well-balanced (no significant overfitting).")
    else:
        print("Warning: Model shows signs of overfitting.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "linear_regression.joblib")
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)

    image_path = os.path.join(IMAGE_DIR, "linear_regression_actual_vs_predicted.png")

    display_metrics(y_test, y_pred, "Linear Regression")
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        "Linear Regression",
        save_path=image_path
    )

    print(f"Model saved to:\n{model_path}")
    print(f"Image saved to:\n{image_path}")

if __name__ == "__main__":
    run_linear_regression()