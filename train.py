"""train.py
Train and save regression pipelines for Immo Eliza project.
Creates and saves: linear_pipeline.joblib, rf_pipeline.joblib, xgb_pipeline.joblib

Usage:
    python train.py

Files read/written (relative to repo root):
    data/raw/dataframe.json -> data/clean/cleaned_data.json
    models/*.joblib
"""

import os
import html
import joblib
import argparse
from typing import List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb

# Paths (relative)
INPUT_FILE = os.path.join("data", "raw", "dataframe.json")
OUTPUT_DIR = os.path.join("data", "clean")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.json")
MODEL_DIR = os.path.join("models")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def dataframe_cleaner(input_path: str, output_path: str) -> pd.DataFrame:
    """Load raw json, drop duplicates, coerce numeric columns, rename, fill defaults and save cleaned file."""
    df = pd.read_json(input_path)

    clean_df = df.drop_duplicates(subset=["postcode", "price", "livable_surface", "address"]).copy()
    clean_df["price_per_m2"] = clean_df["price"] / clean_df["livable_surface"]

    cols_to_numeric = [
        "postcode", "price", "build_year", "bedroom_count", "livable_surface",
        "total_surface", "garage", "terrace", "swimming_pool", "peb_category",
        "Preschool_distance", "Train_station_distance", "Supermarket_distance",
    ]
    for col in cols_to_numeric:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    new_column_names = {
        "peb_category": "energy_consumption_kWh/m2/year",
        "Preschool_distance": "preschool_distance_m",
        "Train_station_distance": "train_station_distance_m",
        "Supermarket_distance": "supermarket_distance_m",
        "distance_nearest_city": "nearest_city_distance_km",
    }
    clean_df = clean_df.rename(columns=new_column_names)

    text_cols = ["city", "province", "address", "nearest_city"]
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

    if "garage" in clean_df.columns:
        clean_df["garage"] = clean_df["garage"].fillna(0)
    if "swimming_pool" in clean_df.columns:
        clean_df["swimming_pool"] = clean_df["swimming_pool"].fillna(0)

    if "property_state" in clean_df.columns:
        clean_df["property_state"] = clean_df["property_state"].replace("To be renovated", "To renovate")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_json(output_path, orient="records", force_ascii=False, indent=4)
    return clean_df


# Small transformer to group rare categories
from sklearn.base import TransformerMixin, BaseEstimator

class TopKCategoricalGrouper(TransformerMixin, BaseEstimator):
    def __init__(self, top_k: int = 50):
        self.top_k = int(top_k)
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X_df.columns:
            # keep the top_k frequent categories
            self.top_categories_[col] = X_df[col].value_counts().nlargest(self.top_k).index.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X_df.columns:
            top = self.top_categories_.get(col, [])
            X_df[col] = X_df[col].where(X_df[col].isin(top), other="__other__")
        return X_df


def make_preprocessor(num_cols: List[str], cat_cols: List[str], strategy: str = "tree", top_k: int = 50):
    """Create a ColumnTransformer. strategy: 'tree' uses OrdinalEncoder; 'linear' uses OneHotEncoder (sparse)."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_group = TopKCategoricalGrouper(top_k=top_k)
    if strategy == "tree":
        cat_pipe = Pipeline([
            ("group", cat_group),
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
    else:
        cat_pipe = Pipeline([
            ("group", cat_group),
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop", sparse_threshold=0.1)

    return preprocessor


def display_metrics(y_true, y_pred, name: str = "Model"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name} Metrics")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: €{rmse:,.2f}")
    print(f"MAE: €{mae:,.2f}\n")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

    print("Cleaning dataframe...")
    df_cleaned = dataframe_cleaner(INPUT_FILE, OUTPUT_FILE)

    # Define feature lists (adapt if columns missing)
    default_num = [
        'latitude','longitude','livable_surface','total_surface','garage','terrace','swimming_pool',
        'energy_consumption_kWh/m2/year','preschool_distance_m','train_station_distance_m',
        'supermarket_distance_m','nearest_city_distance_km','price_per_m2'
    ]
    default_cat = ['property_type','city','province','property_state','nearest_city']

    available = set(df_cleaned.columns)
    num_cols = [c for c in default_num if c in available]
    cat_cols = [c for c in default_cat if c in available]

    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    df_model = df_cleaned.dropna(subset=['price', 'livable_surface']).copy()
    X = df_model[num_cols + cat_cols]
    y = df_model['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    # --- Linear Regression (baseline) ---
    print("\nTraining Linear Regression (baseline) ...")
    preprocessor_linear = make_preprocessor(num_cols, cat_cols, strategy='linear', top_k=50)
    linear_reg = TransformedTargetRegressor(
        regressor=Pipeline([('pre', preprocessor_linear), ('lr', LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1
    )
    cv_scores = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print("Linear CV R2 mean:", cv_scores.mean(), "std:", cv_scores.std())
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    display_metrics(y_test, y_pred, 'Linear Regression')
    joblib.dump(linear_reg, os.path.join(MODEL_DIR, 'linear_pipeline.joblib'))

    # --- Random Forest ---
    print("\nTraining Random Forest ...")
    preprocessor_tree = make_preprocessor(num_cols, cat_cols, strategy='tree', top_k=50)
    rf_reg = TransformedTargetRegressor(
        regressor=Pipeline([('pre', preprocessor_tree), ('rf', RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))]),
        func=np.log1p, inverse_func=np.expm1
    )
    cv_scores_rf = cross_val_score(rf_reg, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
    print("RF CV R2 mean:", cv_scores_rf.mean(), "std:", cv_scores_rf.std())
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    display_metrics(y_test, y_pred_rf, 'Random Forest')
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, 'rf_pipeline.joblib'))

    # --- XGBoost ---
    print("\nTraining XGBoost ...")
    preprocessor_xgb = make_preprocessor(num_cols, cat_cols, strategy='tree', top_k=50)
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
    xgb_pipeline = TransformedTargetRegressor(
        regressor=Pipeline([('pre', preprocessor_xgb), ('xgb', xgb_model)]),
        func=np.log1p, inverse_func=np.expm1
    )
    # Fit without early stopping for simplicity; consider using eval_set + early_stopping_rounds for tuning
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    display_metrics(y_test, y_pred_xgb, 'XGBoost')
    joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, 'xgb_pipeline.joblib'))

    print("\nAll models trained and saved to:", MODEL_DIR)


if __name__ == '__main__':
    main()
