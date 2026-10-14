import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVR
from pathlib import Path


# Paths are derived from this file's location, so the package runs
# from any working directory and from any clone of the repository.
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
IMAGE_DIR = BASE_DIR / "images"


def display_metrics(y_test, y_pred, model_name="Model"):
    """
    Computes and prints standard regression evaluation metrics.
    Includes R², MSE, RMSE, and MAE for model assessment.
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


def plot_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    """
    Plots actual vs predicted values with a reference diagonal line.
    Saves the plot as a PNG file in the images directory.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        color="yellow",
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

    save_path = os.path.join(
        IMAGE_DIR,
        f"{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to:\n{save_path}")

    plt.show()


def run_linearsvr_pipeline(file_path=None):
    """
    Loads dataset, preprocesses features, and trains a Linear SVR model.
    Performs hyperparameter tuning using RandomizedSearchCV.
    Evaluates performance, saves model, and exports visualization.

    file_path defaults to the cleaned dataset so this can be called with no
    argument, like the other model entry points driven by main.py.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_json(file_path or OUTPUT_FILE)
    df = df.dropna(subset=["price"])

    # property_id is a listing number assigned by the website, not a property
    # attribute: as a feature, the model learns which ID ranges were scraped
    # from which search, which says nothing about a new listing.
    X = df.drop(columns=["price", "address", "property_id"], errors="ignore")
    y = df["price"]

    # Target leakage: cleaning.py persists price_per_m2 = price / livable_surface
    # into cleaned_data.json. With livable_surface also a feature, price is
    # recoverable as price_per_m2 * livable_surface, which bought R2 0.91 and a 4%
    # median error by doing arithmetic instead of predicting. Dropped by derivation,
    # not by name, so a future price_* column cannot quietly reintroduce it.
    # See README, "Model comparison", for the measured cost of removing it.
    leaking = [c for c in X.columns if "price" in c.lower()]
    if leaking:
        print(f"Dropping features derived from the target: {leaking}")
    X = X.drop(columns=leaking)

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", preprocessor),
            ("svr", LinearSVR(dual="auto", random_state=42, max_iter=50000))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_dist = {
        "regressor__svr__C": [0.01, 0.1, 1, 10],
        "regressor__svr__epsilon": [0, 0.01, 0.1],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        random_state=42,
        scoring="r2"
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = search.predict(X_test)

    train_pred = search.predict(X_train)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")

    if abs(r2_train - r2_test) < 0.1:
        print("Model is well-balanced (no significant overfitting).")
    else:
        print("Warning: Model shows signs of overfitting.")

    model_path = os.path.join(MODEL_DIR, "linear_svr.joblib")
    joblib.dump(best_model, model_path)

    print(f"Model saved to:\n{model_path}")

    display_metrics(y_test, y_pred, "Linear SVR")
    plot_actual_vs_predicted(y_test, y_pred, "Linear_SVR")
    

if __name__ == "__main__":
    run_linearsvr_pipeline()