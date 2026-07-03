import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"
IMAGE_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\images"


def display_metrics(y_test, y_pred, model_name="Model"):
    """
    Computes regression evaluation metrics for model performance analysis.
    Includes R², MSE, RMSE, and MAE.
    Provides a clear numeric summary of prediction quality.
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
    Adds a reference diagonal line for perfect predictions.
    Optionally saves the plot as a PNG file.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        color="darkblue",
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


def run_xgboost_pipeline():
    """
    Loads dataset, preprocesses features, and trains an XGBoost regression model.
    Applies ordinal encoding for categorical variables and log-transform on target.
    Evaluates performance, saves model, and exports visualization.
    """
    df = pd.read_json(OUTPUT_FILE)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()

    X = df.drop(columns=["price", "address"], errors="ignore")
    y = df["price"]

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "xgboost.joblib")
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")

    image_path = os.path.join(IMAGE_DIR, "xgboost.png")

    display_metrics(y_test, y_pred, "XGBoost Regression")
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        "XGBoost Regression",
        save_path=image_path
    )

    print(f"Image saved to:\n{image_path}")
    print(f"Model saved to:\n{model_path}")


if __name__ == "__main__":
    run_xgboost_pipeline()