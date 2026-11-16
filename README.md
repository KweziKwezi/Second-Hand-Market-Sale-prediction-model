# 🚗 Second-Hand Market Sale Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

A commercially viable machine learning model that predicts the selling price of used cars with a **13.51% average error (MAPE)**. This accuracy makes it highly suitable for real-world dealership pricing, insurance valuation, and buyer/seller estimation tools.

## 📊 Project Overview

This project builds, tunes, and deploys a Decision Tree Regressor to analyze 9 car features (Year, Present Price, Kms Driven, Fuel Type, Seller Type, Transmission, Owner, etc.) and predict the final selling price. The model was rigorously optimized through iterative hyperparameter tuning to minimize prediction error and prevent overfitting.

## 🎯 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 54,491 | Average prediction is off by ~54K currency units |
| **MAPE** | **13.51%** | Average prediction is within ~13.5% of the actual price |
| **Dataset** | 301 listings | 80/20 train-validation split |

> 💡 *A MAPE under 15% is considered **"Very Good"** for commercial used-car pricing models, providing a tight, trustworthy margin for business applications.*

## 🛠️ Tech Stack

- **Python 3.13** — Core programming language
- **Pandas** — Data manipulation, scaling, and preprocessing
- **Scikit-Learn** — Machine learning modeling, evaluation, and train/test splitting
- **Joblib** — Model serialization for production deployment
- **Jupyter Notebook** — Interactive development and experimentation

## 🔬 Methodology

### 1. Data Preprocessing
- **Currency Scaling**: Converted prices from Lakhs (e.g., `3.35`) to full currency values (e.g., `335,000`) for realistic error metric calculation.
- **Feature Engineering**: Applied one-hot encoding to categorical variables (`Car_Name`, `Fuel_Type`, `Selling_type`, `Transmission`), expanding the feature space to **109 total features**.

### 2. Iterative Hyperparameter Tuning
Systematically tested parameter combinations via custom validation loops to find the optimal balance between bias and variance:

| Parameter | Tested Range | Best Value |
|-----------|--------------|------------|
| `max_leaf_nodes` | 5 → 800 | **300** |
| `max_depth` | 1 → 60 | **8** |
| `min_samples_leaf` | 1 → 10 | **1** |
| `min_samples_split` | 2 → 20 | **2** |

### 3. Final Model Configuration
```python
DecisionTreeRegressor(
    max_leaf_nodes=300,
    max_depth=8,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=1
)