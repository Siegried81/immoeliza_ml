import pandas as pd
import numpy as np
import os
import joblib
import gc
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path

# Paths are derived from this file's location, so the package runs
# from any working directory and from any clone of the repository.
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
IMAGE_DIR = BASE_DIR / "images"


def get_preprocessor(num_cols, cat_cols):
    """
    Builds preprocessing pipeline for numerical and categorical features.
    Numerical data is imputed with median and scaled for distance-based learning.
    Categorical data is imputed and one-hot encoded for compatibility with kNN.
    """

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    return ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])


def train_knn(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    """
    Trains a k-Nearest Neighbors regression model with hyperparameter tuning.
    Uses RandomizedSearchCV to optimize number of neighbors and weighting strategy.
    Evaluates model performance and saves both model and visualization.
    """

    print(f"\n{'='*30}\nTraining kNN Regressor\n{'='*30}")

    pipeline = Pipeline([
        ('preprocessor', get_preprocessor(num_cols, cat_cols)),
        ('regressor', KNeighborsRegressor())
    ])

    # Hyperparameter search space for kNN
    param_dist = {
        'regressor__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'regressor__weights': ['uniform', 'distance']
    }

    # Cross-validation search
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Params: {search.best_params_}")
    print(f"R² Score : {r2:.4f} | RMSE : €{rmse:,.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_knn.joblib'))

    # Save performance plot
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color="green")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("kNN: Actual vs Predicted")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.savefig(os.path.join(IMAGE_DIR, "knn_performance.png"))
    plt.close()

    # Cleanup memory
    del best_model, search, pipeline
    gc.collect()

def run_knn():
    """
    Loads the cleaned data, builds the train/test split and runs train_knn.

    Entry point with no argument, like the other model modules, so main.py can
    call every pipeline the same way.
    """
    df = pd.read_json(OUTPUT_FILE).dropna(subset=['price'])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price').tolist()

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
    # High-cardinality columns (city, address) would explode the one-hot matrix
    # and make the distance metric meaningless for kNN.
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() < 50]

    X = df[num_cols + cat_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_knn(X_train, X_test, y_train, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    run_knn()