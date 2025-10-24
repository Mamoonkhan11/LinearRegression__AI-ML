# Task 3 — Linear Regression (House Price Prediction)

A complete **Machine Learning pipeline** using **Linear Regression** to predict house prices based on various property features.  
This project demonstrates data preprocessing, encoding, model training, evaluation, and visualization — following a clean, modular structure.

---

## Overview
This project implements an **end-to-end regression workflow** using Python and Scikit-learn.  
You’ll learn how to:
- Prepare raw housing data for machine learning.
- Handle categorical variables and encode them numerically.
- Train a Linear Regression model.
- Evaluate performance using statistical metrics.
- Visualize results such as correlation heatmaps and residual plots.

---

## Objectives
- Load and clean the housing dataset.  
- Encode non-numeric features (e.g., *furnishingstatus*, *mainroad*).  
- Train a Linear Regression model to predict `price`.  
- Evaluate using **MAE**, **MSE**, and **R²**.  
- Save trained model, metrics, and plots to organized folders.

---

## Project Structure
```
Task3-Linear-Regression/
│
├── data/
│   ├── raw/             
│   └── processed/        
│
├── notebooks/
│   └── eda_and_baseline.ipynb  
│
├── src/
│   ├── data_loading.py        
│   ├── preprocessing.py        
│   ├── features.py             
│   ├── train.py                
│   ├── evaluate.py              
│   └── utils.py                 
│
├── outputs/
│   ├── models/      
│   ├── figures/      
│   └── metrics/       
│
├── requirements.txt
├── main.py                    
├── README.md
└── .gitignore
```

---

## How It Works
1. **Data Loading** — Reads raw CSV and stores a copy in `data/raw/`.
2. **Preprocessing** — Cleans data, encodes categorical columns, and scales numeric features.
3. **Model Training** — Splits data (train/test) and fits a `LinearRegression()` model.
4. **Evaluation** — Calculates MAE, MSE, R²; generates and saves plots.
5. **Outputs** — Saves processed data, model, and evaluation metrics in the `outputs/` folder.

---

## Tools & Libraries
- **Python 3.12+**
- **pandas**, **numpy**
- **scikit-learn**
- **matplotlib**, **seaborn**
- **joblib**

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶ Run the Project
1. Place your dataset (e.g., `Housing.csv`) into `data/raw/`.
2. Run the full pipeline:
   ```bash
   python main.py --input data/raw/Housing.csv --target price
   ```
3. View results in:
   ```
   outputs/
   ├── models/
   ├── metrics/
   └── figures/
   ```

---


## 📈 Evaluation Metrics
| Metric | Description | Ideal Value |
|---------|--------------|--------------|
| **MAE** | Mean Absolute Error | Closer to 0 |
| **MSE** | Mean Squared Error | Lower is better |
| **R²**  | Coefficient of Determination | Closer to 1 |

---

## Learning Outcomes
- Understand how Linear Regression works for numerical prediction.  
- Gain hands-on experience with data preprocessing and encoding.  
- Evaluate regression performance using standard metrics.  
- Build modular, production-style ML projects.
  

---

## References
- Dataset: [Housing Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)  
- Guide: *AI & ML Internship Task 3 — Linear Regression*

---