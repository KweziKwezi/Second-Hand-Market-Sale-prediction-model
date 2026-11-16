# 🚗 Second-Hand Market Sale Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

A commercially viable machine learning model that predicts the selling price of used cars with a **13.51% average error (MAPE)**. This accuracy makes it highly suitable for real-world dealership pricing, insurance valuation, and buyer/seller estimation tools.

---

## 📌 What This Model Is About
This project addresses a common challenge in the automotive industry: accurately estimating the depreciated value of a used vehicle. The model analyzes **9 key attributes** of a car (Year, Present Price, Kilometers Driven, Fuel Type, Seller Type, Transmission, Number of Previous Owners, and Car Name) to predict its final market selling price. 

By leveraging a **Decision Tree Regressor**, the model learns the non-linear relationships and complex interactions between a car's age, usage, and original value to output a highly accurate, data-driven price estimate.

---

## 🎯 Key Results
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 54,491 | Average prediction is off by ~54K currency units |
| **MAPE** | **13.51%** | Average prediction is within ~13.5% of the actual price |
| **Dataset** | 301 listings | 80/20 train-validation split, expanded to 109 features |

---

## 🛠️ Steps Taken for Efficiency & Low Margin of Error
To ensure the model operated efficiently and avoided the common pitfalls of Decision Trees (like severe overfitting), I implemented a rigorous, multi-step pipeline:

1. **Realistic Currency Scaling**: The raw dataset stored prices in "Lakhs" (e.g., `3.35`). I multiplied `Selling_Price` and `Present_Price` by `100,000` to convert them to full currency values (e.g., `335,000`). This ensured that error metrics (MAE/MAPE) reflected real-world financial values, not abstract decimals.
2. **Robust Feature Engineering**: Applied One-Hot Encoding to all categorical variables (`Car_Name`, `Fuel_Type`, `Selling_type`, `Transmission`). This expanded the feature space to **109 distinct features**, allowing the model to capture the unique value impact of specific car models and fuel types without introducing false ordinal relationships.
3. **Sequential Hyperparameter Tuning**: Instead of guessing parameters or using a blind grid search, I performed iterative, sequential tuning. I optimized one parameter at a time, locking in the best value before testing the next. This prevented overfitting while maximizing predictive power.
4. **Full-Dataset Final Training**: After validating the optimal parameters on the 80/20 split, I retrained the final model on **100% of the dataset**. This ensures the model learns from every single available data point, maximizing its generalization capability for production use.

---

## 🔬 The Hyperparameter Tuning Process
Decision Trees are prone to memorizing training data. To force the model to learn *general patterns* rather than *noise*, I systematically tested and optimized four critical parameters:

| Tuning Step | Parameter Tested | Range Tested | Best Value Found | Resulting MAE |
|-------------|------------------|--------------|------------------|---------------|
| **1. Tree Size** | `max_leaf_nodes` | 5 → 800 | **300** | ~58,723 |
| **2. Leaf Stability** | `min_samples_leaf` | 1 → 10 | **1** | ~58,723 |
| **3. Depth Control** | `max_depth` | 1 → 60, None | **8** | ~54,490 |
| **4. Split Threshold**| `min_samples_split`| 2 → 20 | **2** | **54,491** |

**Final Winning Configuration:**
```python
DecisionTreeRegressor(
    max_leaf_nodes=300,
    max_depth=8,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=1
)
