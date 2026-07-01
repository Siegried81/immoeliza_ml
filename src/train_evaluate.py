import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def plot_regression_results(X_test, y_test, y_pred, feature_name):
    """Visualise les prix réels par rapport aux prix prédits."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature_name], y_test, color='blue', alpha=0.5, label='Actual prices')
    plt.scatter(X_test[feature_name], y_pred, color='red', alpha=0.5, label='Predicted prices')
    plt.title(f'Random Forest: Price vs {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def run_training():
    # Chemins d'accès
    data_path = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\processed\processed_data.csv"
    model_dir = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\models"
    model_file = os.path.join(model_dir, "model.joblib")
    
    # Chargement des données
    df_final = pd.read_csv(data_path)
    
    # Split
    X = df_final.drop(columns=['price'])
    y = df_final['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement (Random Forest au lieu de la régression linéaire)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, y_pred):.2f}")
    
    # Sauvegarde
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")
    
    # Visualisation
    if 'livable_surface' in X_test.columns:
        plot_regression_results(X_test, y_test, y_pred, 'livable_surface')

if __name__ == "__main__":
    run_training()