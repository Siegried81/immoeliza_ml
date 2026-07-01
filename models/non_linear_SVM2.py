import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'clean'
OUTPUT_FILE = OUTPUT_DIR / 'cleaned_data.json'
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="blue", edgecolors="black")
    line_min, line_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def run_non_linear_svr():
    # Load and prepare data
    df = pd.read_json(OUTPUT_FILE).dropna(subset=["price"])
    
    # Using 50% of data to speed up the  RBF SVR
    df = df.sample(frac=0.5, random_state=42)
    print(f"Training on {len(df)} samples.")
    
    X = df.drop(columns=["price", "address"], errors="ignore")
    y = df["price"]
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])
    
    # PIPELINE: Includes SVD to compress feature space, keeping RBF training feasible
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('svd', TruncatedSVD(n_components=100)),
        ('svr', SVR(kernel='rbf'))
    ])
    
    param_grid = {
        'svr__C': [100, 1000, 5000],
        'svr__gamma': ['scale', 0.1, 0.01]
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomizedSearchCV with progress tracking
    grid = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=2 
    )
    
    print("Starting optimized non-linear training...")
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, "optimized_non_linear_svr.joblib"))
    
    y_pred = best_model.predict(X_test)
    display_metrics(y_test, y_pred, "Non-Linear SVR (RBF + SVD)")
    plot_actual_vs_predicted(y_test, y_pred, "Non-Linear SVR (RBF + SVD)")

if __name__ == "__main__":
    run_non_linear_svr()