import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                     ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                                     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer(transformers=[('num', num_transformer, num_cols), 
                                           ('cat', cat_transformer, cat_cols)])

def run_preprocessing():
    df = pd.read_json(r"D:\Users\Siegried\Desktop\Becode\immoeliza_ml\data\clean\cleaned_data.json")
    
    num_cols = ['postcode', 'latitude', 'longitude', 'price', 'build_year', 'bedroom_count', 
                'livable_surface', 'total_surface', 'garage', 'terrace', 'swimming_pool', 
                'energy_consumption_kWh/m2/year', 'preschool_distance_m', 'train_station_distance_m', 
                'supermarket_distance_m', 'nearest_city_distance_km']
    cat_cols = ['property_type', 'city', 'province', 'property_state', 'nearest_city']
    
    # Analyze only numerical features
    df_num = df[num_cols]
    corr_matrix = df_num.corr().abs()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Numerical Feature Correlation")
    plt.show()

    # Identify redundant numerical columns (threshold = 0.8)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"Dropping highly correlated numerical features: {to_drop}")
    
    # Process data
    filtered_num_cols = [c for c in num_cols if c not in to_drop]
    preprocessor = get_preprocessor(filtered_num_cols, cat_cols)
    
    X_processed = preprocessor.fit_transform(df[filtered_num_cols + cat_cols])
    
    # Reconstruct df
    cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
    all_feature_names = filtered_num_cols + list(cat_feature_names)
    
    return pd.DataFrame(X_processed, columns=all_feature_names)

if __name__ == "__main__":
    final_df = run_preprocessing()
    print(f"Final shape: {final_df.shape}")