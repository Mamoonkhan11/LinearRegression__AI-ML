# Loading data and saving a raw copy

import os
import pandas as pd

def Load_data(input_path: str, save_raw: bool = True, raw_path: str = "Data/raw/dataset_raw.csv") -> pd.DataFrame:
    
    df = pd.read_csv(input_path)
    print(f"[data_loading] Loaded data from {input_path} with shape {df.shape}")
    if save_raw:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"[data_loading] Raw copy saved to {raw_path}")
    return df