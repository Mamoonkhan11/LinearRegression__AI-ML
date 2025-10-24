# Training script for Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.utils import Save_model
import os

def Train_model(X, y, model_output_path="outputs/models/linear_regression.pkl", test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    Save_model(model, model_output_path)
    print(f"[train] Model trained and saved to {model_output_path}")
    return model, X_train, X_test, y_train, y_test
