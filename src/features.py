# Features extraction and preprocessing functions

import pandas as pd

def default_feature_selector(df: pd.DataFrame, target_column: str):
  
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column in numeric:
        numeric.remove(target_column)

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical