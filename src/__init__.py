"""
Public surface of the src package.

Only modules that live in src/ are re-exported here. The model entry points
live in models/ and are imported from there (see main.py); listing them here
made `import src` fail, which broke every module that imports from src.
"""
from .cleaning import dataframe_cleaner
from .preprocessing import get_preprocessor, remove_highly_correlated_features
from .plot_missing import plot_missing_values
from .heatmap import run_preprocessing as run_heatmap_analysis
from .overfitting_XGBoost import run_xgboost_pipeline, check_overfitting
from .predict import run_prediction
