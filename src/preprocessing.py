# Preprocessing functions for data cleaning and transformation

import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def Get_default_preprocessor(numeric_features, categorical_features):
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    return preprocessor

def Preprocess(df: pd.DataFrame,
               target_column: str,
               numeric_features: list,
               categorical_features: list,
               save_processed: bool = True,
               processed_path: str = "Data/processed/Housing.csv"):
    
    df = df.copy()
    # Separate features and target
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    preprocessor = Get_default_preprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    # Get feature names after transformation
    num_cols = numeric_features
    # Handle categorical feature names
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = ohe.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_cols = []

    columns = num_cols + cat_cols
    import numpy as np
    X_df = pd.DataFrame(X_transformed, columns=columns, index=X.index)

    if save_processed:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        pd.concat([X_df, y.reset_index(drop=True)], axis=1).to_csv(processed_path, index=False)
        print(f"\n [preprocessing] Processed dataset saved to {processed_path} \n")

    return X_df, y, preprocessor