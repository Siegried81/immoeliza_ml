import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
PLOT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\missing_values_plot.png"

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