import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.linear_regression import run_linear_regression
from models.Random_Forest import run_random_forest
from models.linear_svr import run_linearsvr_pipeline
from models.non_linear_svr import run_non_linear_svr
from models.XGBoost import run_xgboost_pipeline
from models.tuning_XGBoost import run as run_xgboost_tuning
from models.cross_validation_kNN_XGBoost import run_knn
from src import paths

def main():
    """The menu over the seven model scripts.

    Each script builds its own preprocessor from the cleaned data, so the menu
    passes nothing: it used to build one here and hand it to functions that
    take no argument, one of which was imported under a name that did not
    exist, so no option could run.
    """
    print("IMMOELIZA ML PREDICTION SYSTEM")

    menu = {
        "1": ("Linear Regression", run_linear_regression),
        "2": ("Random Forest", run_random_forest),
        "3": ("Linear SVR", lambda: run_linearsvr_pipeline(paths.CLEAN_DATA)),
        "4": ("Non-Linear SVR", run_non_linear_svr),
        "5": ("XGBoost", run_xgboost_pipeline),
        "6": ("XGBoost Tuned", run_xgboost_tuning),
        "7": ("Cross Validation kNN", run_knn),
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