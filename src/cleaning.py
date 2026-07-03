import pandas as pd
import html
import os

INPUT_FILE = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\raw\dataframe.json"
OUTPUT_DIR = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.json")

def dataframe_cleaner(input_path, output_path):
    """
    Cleans the raw dataframe and saves it as a json file.
    """
    df = pd.read_json(input_path)
    
    clean_df = df.drop_duplicates(subset=['postcode', 'price', 'livable_surface', 'address']).copy()
    clean_df['price_per_m2'] = clean_df['price'] / clean_df['livable_surface']

    cols_to_int = ["postcode", "price", "build_year", "bedroom_count", "livable_surface", 
                   "total_surface", "garage", "terrace", "swimming_pool", "peb_category", 
                   "Preschool_distance", "Train_station_distance", "Supermarket_distance"]

    for col in cols_to_int:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').astype("Int64")
    
    new_column_names = {"peb_category": "energy_consumption_kWh/m2/year",
                        "Preschool_distance": "preschool_distance_m",
                        "Train_station_distance": "train_station_distance_m",
                        "Supermarket_distance": "supermarket_distance_m",
                        "distance_nearest_city": "nearest_city_distance_km"}
    
    clean_df = clean_df.rename(columns=new_column_names)

    text_cols = ['city', 'province', 'address', 'nearest_city']
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

    clean_df.fillna(value={'garage': 0, 'swimming_pool': 0}, inplace=True)

    if "property_state" in clean_df.columns:
        clean_df["property_state"] = clean_df["property_state"].replace("To be renovated", "To renovate")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    clean_df.to_json(output_path, orient="records", force_ascii=False, indent=4)
    
    return clean_df

if __name__ == "__main__":
    df_cleaned = dataframe_cleaner(INPUT_FILE, OUTPUT_FILE)