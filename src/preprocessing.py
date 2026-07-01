import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                     ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                                     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer(transformers=[('num', num_transformer, num_cols), 
                                           ('cat', cat_transformer, cat_cols)])

def remove_highly_correlated_features(df, threshold=0.90):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def run_preprocessing():
    # Define paths
    raw_data_path = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json"
    output_dir = r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_json(raw_data_path)
    num_cols = ['postcode', 'latitude', 'longitude', 'price', 'build_year', 'bedroom_count', 
                'livable_surface', 'total_surface', 'garage', 'terrace', 'swimming_pool', 
                'energy_consumption_kWh/m2/year', 'preschool_distance_m', 'train_station_distance_m', 
                'supermarket_distance_m', 'nearest_city_distance_km']
    cat_cols = ['property_type', 'city', 'province', 'property_state', 'nearest_city']
    
    # Preprocess
    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_processed = preprocessor.fit_transform(df[num_cols + cat_cols])
    
    feature_names = num_cols + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols))
    df_results = pd.DataFrame(X_processed, columns=feature_names)
    
    # Filters
    # 1. Target correlation filter (>= 0.05)
    df_results = df_results[df_results.corr().abs()['price'][df_results.corr().abs()['price'] >= 0.05].index]
    # 2. Multicollinearity filter
    df_final = remove_highly_correlated_features(df_results, threshold=0.90)
    
    # Save
    df_final.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)
    print(f"Preprocessing complete. Data saved to {output_dir}")

if __name__ == "__main__":
    run_preprocessing() 