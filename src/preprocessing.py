import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# This module only builds reusable transformers: it reads and writes no file,
# so it needs no path constant. train.py owns the data paths.

def get_preprocessor(num_cols, cat_cols):
    """
    Creates a scikit-learn ColumnTransformer that applies median imputation and
    standard scaling to numerical features, and constant imputation followed
    by one-hot encoding to categorical features.

    It is returned unfitted on purpose: callers put it inside a Pipeline so the
    imputation medians and the scaler are fitted on the training split only.
    """
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

def remove_highly_correlated_features(df, threshold=0.90):
    """
    Identifies and removes features from a DataFrame that exhibit multicollinearity
    above a specified correlation threshold.

    It compares features with each other only, never with the target, so it is
    safe to call on a training split without leaking the target.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
