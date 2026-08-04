# PEMFC Voltage Loss & Degradation Forecasting using XGBoost & SHAP

This repository presents a machine learning framework for predicting dynamic **voltage loss (%) and degradation** in Proton Exchange Membrane Fuel Cells (PEMFCs). 

The framework consists of two main stages:
1. **Explainable Feature Selection (SHAP):** Screening physical sensor channels (temperatures, pressures, flow rates) and dynamic lag features using Tree SHAP values.

2. **Recursive Walk-Forward Forecasting:** A multi-step autoregressive XGBoost model featuring periodic 24-hour re-anchoring to simulate real-world operational health monitoring.

3. ## 🧬 Background & Physical Baseline

Rather than predicting raw stack output voltage directly, the degradation target is normalized against a baseline polarization curve to account for dynamic current operating points:

To eliminate non-stationary drift and focus on operational degradation drivers, the model is trained on the hourly first-difference:

## ⚙️ Key Features & Methodology

### 1. Feature Importance via SHAP
* Generates lag,rolling window statistics and target differences across all physical channels.
* Uses `shap.TreeExplainer` to evaluate global feature contribution on training data.
* Aggregates fine-grained lag features back to their primary physical sensor sources (e.g., Temperature, Current).

### 2. Walk-Forward Prediction
* **Recursive Autoregression:** Uses predicted target deltas as lagged inputs during multi-step execution.
* **Exogenous Persistence Modeling:** Assumes zero-order hold for operational variables during multi-step horizons.
* **Periodic 24-Hour Re-Anchoring:** Resets accumulated numerical integration drift back to ground-truth physical state measurements every 24 operating hours, modeling periodic maintenance diagnostic routines.

* ## 📊 Evaluation & Metrics

The model is evaluated on an unseen out-of-sample test split (20%) using:
* **MAE:** Mean Absolute Error on reconstructed voltage loss (%)
* **RMSE:** Root Mean Squared Error
* **$R^2$ Score:** Coefficient of Determination

* ## 🛠️ Stack
* **Python 3.9+**
* **XGBoost:** Gradient boosting engine
* **SHAP:** Model explainability & feature relevance attribution
* **Pandas & NumPy:** Time-series feature engineering & integration
* **Matplotlib:** Visualization
