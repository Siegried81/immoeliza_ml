import pandas as pd
import numpy as np
import os
import joblib
import gc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
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
    Handles missing values, scaling, and encoding.
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


# Training XGBoost
def train_xgboost(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    """
    Trains XGBoost model with hyperparameter tuning.
    Evaluates performance and saves model + plot.
    """
    print("Training XGBoost...")

    pipeline = Pipeline([
        ('preprocessor', get_preprocessor(num_cols, cat_cols)),
        ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    param_dist = {
        'regressor__n_estimators': [100, 200, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7, 9, 11],
        'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics, all on the euro scale: this script predicts price directly, so
    # no inverse transform is needed (src/train.py trains on log1p and has to
    # expm1 before measuring). MAPE is reported too, because a percentage is
    # what a reader of these numbers actually compares across models - and the
    # median absolute percentage error beside it, since MAPE is pulled up by
    # cheap properties where a small euro error is a large share of the price.
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    ape = np.abs(np.asarray(y_test) - y_pred) / np.asarray(y_test) * 100
    n_test = len(y_pred)

    print(f"Best Params: {search.best_params_}")
    print(f"Held-out rows : {n_test:,}")
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:,.2f}")
    print(f"RMSE     : {rmse:,.2f}")
    print(f"MAE      : {mae:,.2f}")
    print(f"MAPE     : {ape.mean():.2f} %  (median {np.median(ape):.2f} %)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Its own file: src/train.py writes best_xgboost.joblib, and the two fit
    # different models (that one trains on log1p(price) and keeps `city`, this
    # one trains on raw euros and drops every category with 50+ values). Sharing
    # the path meant whichever ran last silently became "the" model, and the
    # metrics quoted in the README depended on that order.
    model_path = os.path.join(MODEL_DIR, "best_xgboost_tuned.joblib")
    joblib.dump(best_model, model_path)

    image_path = os.path.join(IMAGE_DIR, "xgboost_performance.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color="purple")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--")

    plt.title("XGBoost: Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Model saved to: {model_path}")
    print(f"Image saved to: {image_path}")

    del best_model, search, pipeline
    gc.collect()

def run():
    """
    Loads data, prepares features, and runs training pipeline.
    """
    df = pd.read_json(OUTPUT_FILE).dropna(subset=['price'])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price').tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Target leakage: cleaning.py persists price_per_m2 = price / livable_surface
    # into cleaned_data.json, and select_dtypes picks up every numeric column, so
    # this script was handing the model a column computed FROM the answer. With
    # livable_surface also a feature, price is recoverable as
    # price_per_m2 * livable_surface. That is what produced R2 0.91 and a 4%
    # median error against the honest pipeline's 0.77 and 15%: the score measured
    # arithmetic, not prediction. On a real listing price_per_m2 is unknowable
    # without the price. src/train.py never had this because it names its columns.
    # Dropped by derivation, not by name, so a future price_* column cannot
    # quietly reintroduce it.
    leaking = [c for c in num_cols if 'price' in c.lower()]
    if leaking:
        print(f"Dropping features derived from the target: {leaking}")
    num_cols = [c for c in num_cols if c not in leaking]

    cat_cols = [c for c in cat_cols if df[c].nunique() < 50]

    print(f"Using {len(cat_cols)} categorical columns after filtering high cardinality.")

    X = df[num_cols + cat_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_xgboost(X_train, X_test, y_train, y_test, num_cols, cat_cols)

if __name__ == "__main__":
    run()