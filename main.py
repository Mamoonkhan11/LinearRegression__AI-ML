# Main file to test the pipeline

import argparse
from src.utils import Set_seed
from src.data_loading import Load_data
from src.features import Default_feature_selector
from src.preprocessing import Preprocess
from src.train import Train_model
from src.evaluate import Evaluate_and_save

def run_pipeline(input_csv: str, target: str):
    Set_seed(42)
    df = Load_data(input_csv, save_raw=True, raw_path="data/raw/dataset_raw.csv")
    numeric, categorical = Default_feature_selector(df, target)
    print(f"[main] Numeric features detected: {numeric}")
    print(f"[main] Categorical features detected: {categorical}")
    X, y, preprocessor = Preprocess(df, target_column=target,
                                    numeric_features=numeric,
                                    categorical_features=categorical,
                                    save_processed=True,
                                    processed_path="data/processed/dataset_processed.csv")
    model, X_train, X_test, y_train, y_test = Train_model(X, y, model_output_path="outputs/models/linear_regression.pkl")
    metrics = Evaluate_and_save(model, X_test, y_test, outputs_dir="outputs")
    print("[main] Pipeline finished. Check outputs/ for models, figures and metrics.")
    print(f"[main] Metrics summary: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Linear Regression pipeline")
    parser.add_argument("--input", type=str, default="data/raw/dataset_raw.csv", help="Path to input CSV")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    args = parser.parse_args()
    run_pipeline(args.input, args.target)