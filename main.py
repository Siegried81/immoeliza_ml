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

def main():
    """
    Interactive menu running one model pipeline per choice.

    Every pipeline is self-contained: it loads the cleaned dataset and builds
    the preprocessing its own feature selection needs (one-hot for the
    distance-based models, ordinal for the tree-based ones), so the menu passes
    no shared preprocessor. Failures are deliberately not caught: a broken
    pipeline must show its traceback instead of a one-line message.
    """
    print("IMMOELIZA ML PREDICTION SYSTEM")

    menu = {
        "1": ("Linear Regression", run_linear_regression),
        "2": ("Random Forest", run_random_forest),
        "3": ("Linear SVR", run_linearsvr_pipeline),
        "4": ("Non-Linear SVR", run_non_linear_svr),
        "5": ("XGBoost", run_xgboost_pipeline),
        "6": ("XGBoost Tuned", run_xgboost_tuning),
        "7": ("Cross Validation kNN", run_knn)
    }

    for key, (name, _) in menu.items():
        print(f"{key} : {name}")
    print("8 : Exit")

    while True:
        choice = input("Choose an option (1-8): ")

        if choice in menu:
            name, function = menu[choice]
            print(f"Running: {name}...")
            function()
        elif choice == "8":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid option. Please choose between 1 and 8.")

if __name__ == "__main__":
    main()
