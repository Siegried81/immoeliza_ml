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

OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"
IMAGE_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\images"


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

if __name__ == "__main__":
    df = pd.read_json(OUTPUT_FILE).dropna(subset=['price'])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price').tolist()
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() < 50]

    X = df[num_cols + cat_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_knn(X_train, X_test, y_train, y_test, num_cols, cat_cols)