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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define paths relative to the script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'clean'
OUTPUT_FILE = OUTPUT_DIR / 'cleaned_data.json'
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def display_metrics(y_test, y_pred, model_name="Model"):
    """Display regression performance metrics."""
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
    """Plot actual prices versus predicted prices."""
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
    X = df.drop(columns=["price", "address"], errors="ignore")
    y = df["price"]
    
    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    # Create Preprocessor
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
    
    # Define Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('svr', SVR(kernel='rbf'))
    ])
    
    # Hyperparameter search space
    param_grid = {
        'svr__C': [100, 1000, 5000],
        'svr__gamma': ['scale', 0.1, 0.01]
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomizedSearchCV for faster execution
    grid = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=2 
    )
    
    grid.fit(X_train, y_train)
    
    # Evaluation and saving
    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, "optimized_svr_all_features.joblib"))
    
    y_pred = best_model.predict(X_test)
    
    display_metrics(y_test, y_pred, "Optimized SVR (RBF)")
    plot_actual_vs_predicted(y_test, y_pred, "Optimized SVR (RBF)")

if __name__ == "__main__":
    run_non_linear_svr()