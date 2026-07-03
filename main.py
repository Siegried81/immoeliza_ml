import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.linear_regression import run_simple_regression
from models.Ramdom_Forest import run_random_forest as run_rf
from models.linear_svr import run_linearsvr_pipeline
from models.non_linear_svr import run_non_linear_svr
from models.XGBoost import run_xgboost_pipeline
from models.tuning_XGBoost import run as run_xgboost_tuning
from models.cross_validation_kNN_XGBoost import train_knn
from src.preprocessing import get_preprocessor

def main():
    num_cols = ['postcode', 'latitude', 'longitude', 'build_year', 'bedroom_count', 
                'livable_surface', 'total_surface', 'garage', 'terrace', 'swimming_pool', 
                'energy_consumption_kWh/m2/year', 'preschool_distance_m', 'train_station_distance_m', 
                'supermarket_distance_m', 'nearest_city_distance_km']
    cat_cols = ['property_type', 'city', 'province', 'property_state', 'nearest_city']
    
    preprocessor = get_preprocessor(num_cols, cat_cols)
    
    print("IMMOELIZA ML PREDICTION SYSTEM")
    
    menu = {
        "1": ("Linear Regression", lambda: run_simple_regression(preprocessor)),
        "2": ("Random Forest", lambda: run_rf(preprocessor)),
        "3": ("Linear SVR", lambda: run_linearsvr_pipeline(preprocessor)),
        "4": ("Non-Linear SVR", lambda: run_non_linear_svr(preprocessor)),
        "5": ("XGBoost", lambda: run_xgboost_pipeline(preprocessor)),
        "6": ("XGBoost Tuned", lambda: run_xgboost_tuning(preprocessor)),
        "7": ("Cross Validation kNN", lambda: train_knn(preprocessor))
    }

    for key, (name, _) in menu.items():
        print(f"{key} : {name}")
    print("8 : Exit")

    while True:
        choice = input("Choose an option (1-8): ")

        if choice in menu:
            name, function = menu[choice]
            print(f"Running: {name}...")
            try:
                function()
            except Exception as e:
                print(f"Error during execution: {e}")
        elif choice == "8":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid option. Please choose between 1 and 8.")

if __name__ == "__main__":
    main()