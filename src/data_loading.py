# Loading data and saving a raw copy

import os
import pandas as pd

def Load_data(input_path: str, save_raw: bool = True, raw_path: str = "Data/raw/Housing.csv") -> pd.DataFrame:
    
    df = pd.read_csv(input_path)
    print(f"\n [data_loading] Loaded data from {input_path} with shape {df.shape} \n ")
    if save_raw:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"\n [data_loading] Raw copy saved to {raw_path} \n")
    return df