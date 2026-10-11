import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
# The repository root on sys.path, so `src.paths` resolves whether this file is
# run through main.py or on its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

OUTPUT_FILE = paths.CLEAN_DATA
PLOT_FILE = paths.MISSING_VALUES_PLOT

def plot_missing_values(file_path):
    df = pd.read_json(file_path)
    total_rows = len(df)
    
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if not nan_cols.empty:
        plt.figure(figsize=(12, 6))
        bars = nan_cols.plot(kind="bar", color="yellow", edgecolor="black")
        
        for i in bars.patches:
            percentage = f'{(i.get_height() / total_rows) * 100:.1f}%'
            plt.text(i.get_x() + i.get_width() / 2, i.get_height(), percentage, 
                     ha='center', va='bottom', fontsize=10)

        plt.title("Columns with Missing Values")
        plt.xlabel("Columns")
        plt.ylabel("Number of missing values")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(PLOT_FILE)
        print(f"Plot saved to: {PLOT_FILE}")
        plt.show()
    else:
        print("No missing values found.")

if __name__ == "__main__":
    plot_missing_values(OUTPUT_FILE)