import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD

OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"
IMAGE_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\images"

def display_metrics(y_test, y_pred, model_name="Model"):
    r2_test = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Metrics")
    print("-" * 40)
    print(f"R² Test Score : {r2_test:.4f}")
    print(f"MSE           : {mse:,.2f}")
    print(f"RMSE          : €{rmse:,.2f}")
    print(f"MAE           : €{mae:,.2f}")

def plot_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    os.makedirs(IMAGE_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        color="blue",
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

    save_path = os.path.join(IMAGE_DIR, "non_linear_svr.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Non Linear SVR
def run_non_linear_svr():
    df = pd.read_json(OUTPUT_FILE).dropna(subset=["price"])

    X = df.drop(columns=["price", "address"], errors="ignore")
    y = df["price"]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

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

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("svd", TruncatedSVD(n_components=100)),
        ("svr", SVR(kernel="rbf"))
    ])

    param_grid = {
        "svr__C": [100, 1000, 5000],
        "svr__gamma": ["scale", 0.1, 0.01]
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "non_linear_svr.joblib")
    joblib.dump(best_model, model_path)

    print(f"Model saved to:\n{model_path}")

    y_pred = best_model.predict(X_test)

    display_metrics(y_test, y_pred, "Non-Linear SVR (RBF + SVD)")

    plot_actual_vs_predicted(y_test, y_pred, "Non-Linear SVR (RBF + SVD)")

if __name__ == "__main__":
    run_non_linear_svr()