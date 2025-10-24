# Main file to test the pipeline

import argparse
from src.utils import set_seed
from src.data_loading import load_data
from src.features import default_feature_selector
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_and_save

def run_pipeline(input_csv: str, target: str):
    set_seed(42)
    df = load_data(input_csv, save_raw=True, raw_path="data/raw/dataset_raw.csv")
    numeric, categorical = default_feature_selector(df, target)
    print(f"[main] Numeric features detected: {numeric}")
    print(f"[main] Categorical features detected: {categorical}")
    X, y, preprocessor = preprocess(df, target_column=target,
                                    numeric_features=numeric,
                                    categorical_features=categorical,
                                    save_processed=True,
                                    processed_path="data/processed/dataset_processed.csv")
    model, X_train, X_test, y_train, y_test = train_model(X, y, model_output_path="outputs/models/linear_regression.pkl")
    metrics = evaluate_and_save(model, X_test, y_test, outputs_dir="outputs")
    print("[main] Pipeline finished. Check outputs/ for models, figures and metrics.")
    print(f"[main] Metrics summary: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Linear Regression pipeline")
    parser.add_argument("--input", type=str, default="data/raw/dataset_raw.csv", help="Path to input CSV")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    args = parser.parse_args()
    run_pipeline(args.input, args.target)