import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

# Paths are derived from this file's location, so the package runs
# from any working directory and from any clone of the repository.
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
IMAGE_DIR = BASE_DIR / "images"


def display_metrics(y_test, y_pred, model_name="Model"):
    """
    Computes and prints regression evaluation metrics.
    Includes R², MSE, RMSE, and MAE.
    Used to assess overall model performance on test data.
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
    Creates a scatter plot comparing actual vs predicted values.
    Includes a perfect prediction reference line.
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


def get_preprocessor(num, cat):
    """
    Builds preprocessing pipeline for numerical and categorical features.
    Handles missing values, scaling, and one-hot encoding.
    Returns a ColumnTransformer ready for model integration.
    """
    num_t = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_t = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_t, num),
        ("cat", cat_t, cat)
    ])


def run_random_forest():
    """
    Trains a Random Forest regression model using cleaned real estate data.
    Performs preprocessing, training, evaluation, and overfitting check.
    Saves trained model and performance visualization.
    """
    df = pd.read_json(OUTPUT_FILE)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()

    cat_cols = ["property_type", "city", "province", "property_state", "nearest_city"]
    num_cols = df.select_dtypes(include=[np.number]).columns.drop("price").tolist()

    # Target leakage: cleaning.py persists price_per_m2 = price / livable_surface
    # into cleaned_data.json. With livable_surface also a feature, price is
    # recoverable as price_per_m2 * livable_surface, which bought R2 0.91 and a 4%
    # median error by doing arithmetic instead of predicting. Dropped by derivation,
    # not by name, so a future price_* column cannot quietly reintroduce it.
    # See README, "Model comparison", for the measured cost of removing it.
    leaking = [c for c in num_cols if "price" in c.lower()]
    if leaking:
        print(f"Dropping features derived from the target: {leaking}")
    num_cols = [c for c in num_cols if c not in leaking]

    X = df[num_cols + cat_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", get_preprocessor(num_cols, cat_cols)),
            ("rf", RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        func=np.log,
        inverse_func=np.exp
    )

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

    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(model, model_path)

    print(f"Model saved to:\n{model_path}")

    y_pred = model.predict(X_test)

    image_path = os.path.join(IMAGE_DIR, "random_forest.png")

    display_metrics(y_test, y_pred, "Random Forest")
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        "Random Forest",
        save_path=image_path
    )

    print(f"Image saved to:\n{image_path}")

if __name__ == "__main__":
    run_random_forest()