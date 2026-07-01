import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVR

# Configuration
OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
MODEL_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"

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
    plt.scatter(y_test, y_pred, alpha=0.3, color="purple", edgecolors="black")
    line_min, line_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(f"{model_name}: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def run_linearsvr_pipeline(file_path):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    df = pd.read_json(file_path)
    target = "price"
    df = df.dropna(subset=[target])
    
    X = df.drop(columns=[target, "address"], errors="ignore")
    y = df[target]

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

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

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('prep', preprocessor),
            ('svr', LinearSVR(dual="auto", random_state=42, max_iter=10000))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        "regressor__svr__C": [0.01, 0.1, 1, 10],
        "regressor__svr__epsilon": [0, 0.01, 0.1],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1, # Utilise tous les cœurs pour accélérer
        random_state=42,
        scoring='r2'
    )

    search.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, 'linearsvr_model.joblib')
    joblib.dump(search.best_estimator_, model_path)
    
    y_pred = search.predict(X_test)
    
    display_metrics(y_test, y_pred, "Linear SVM")
    plot_actual_vs_predicted(y_test, y_pred, "Linear SVM")

if __name__ == "__main__":
    run_linearsvr_pipeline(OUTPUT_FILE)