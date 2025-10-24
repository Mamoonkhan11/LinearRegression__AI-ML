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
    df = Load_data(input_csv, save_raw=True, raw_path="Data/raw/Housing.csv")
    numeric, categorical = Default_feature_selector(df, target)
    print(f"\n [main] Numeric features detected: {numeric} \n")
    print(f"\n [main] Categorical features detected: {categorical} \n")
    X, y, preprocessor = Preprocess(df, target_column=target,
                                    numeric_features=numeric,
                                    categorical_features=categorical,
                                    save_processed=True,
                                    processed_path="Data/processed/Housing.csv")
    model, X_train, X_test, y_train, y_test = Train_model(X, y, model_output_path="outputs/models/linear_regression.pkl")
    metrics = Evaluate_and_save(model, X_test, y_test, outputs_dir="outputs")
    print("\n [main] Pipeline finished. Check outputs/ for models, figures and metrics.\n")
    print(f"\n [main] Metrics summary: {metrics} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Linear Regression pipeline")
    parser.add_argument("--input", type=str, default="Data/raw/Housing.csv", help="Path to input CSV")
    parser.add_argument('--target', type=str, default='price', help='Target value')
    # Default arguments can be modified as needed
    args = parser.parse_args()
    run_pipeline(args.input, args.target)