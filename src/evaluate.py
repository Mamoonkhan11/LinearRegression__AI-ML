# Evaluation Module

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from src.utils import Save_json

def Evaluate_and_save(model, X_test, y_test, outputs_dir="outputs"):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    os.makedirs(os.path.join(outputs_dir, "metrics"), exist_ok=True)
    Save_json(metrics, os.path.join(outputs_dir, "metrics", "metrics.json"))
    print(f"\n [evaluate] Metrics saved to {os.path.join(outputs_dir, 'metrics', 'metrics.json')} \n")

    # Plot: Actual vs Predicted
    os.makedirs(os.path.join(outputs_dir, "figures"), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "figures", "actual_vs_predicted.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\n [evaluate] Actual vs Predicted plot saved to {out_path} \n")

    # Plot: Residuals Distribution
    residuals = y_test - preds
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.tight_layout()
    res_path = os.path.join(outputs_dir, "figures", "residuals_distribution.png")
    plt.savefig(res_path)
    plt.close()
    print(f"\n [evaluate] Residuals plot saved to {res_path} \n")

    return metrics
