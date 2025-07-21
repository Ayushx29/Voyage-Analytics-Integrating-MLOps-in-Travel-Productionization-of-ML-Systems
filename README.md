# Voyage-Analytics-Integrating-MLOps-in-Travel-Productionization-of-ML-Systems

This project integrates robust **Machine Learning** and simulated **MLOps** practices to build and automate end-to-end ML pipelines tailored for the **Travel Industry**. It includes model training, evaluation, deployment, and orchestration across three key problem statements:

---

## 🧩 Key Modules

1. **🧑 Gender Classification** based on user profiles  
2. **🏨 Hotel Recommendation** based on budget and place  
3. **💸 Flight Price Prediction** using booking parameters  

All modules are executed in Google Colab and Docker environments, with model serialization and experiment tracking simulated using **MLflow**, **Airflow**, and **Python-based logging**.

---
## 🧠 Module Breakdown

### 1. 🧑 Gender Classification
- **Inputs**: User Code, Company, Name, Age  
- **Feature Engineering**: PCA on SentenceTransformer embeddings  
- **Model**: Tuned Logistic Regression  
- **Output**: Predicted Gender (Male/Female)

### 2. 🏨 Hotel Recommendation
- **Inputs**: Name, Budget, Place  
- **Model**: Decision Trees predicting hotel name, place, and price  
- **Output**: Hotel suggestion with estimated price

### 3. 💸 Flight Price Prediction
- **Inputs**: Airline, Source, Destination, Date, Stops  
- **Models**: Linear, Ridge, Lasso, ElasticNet, XGBoost, Random Forest  
- **Output**: Predicted flight fare in ₹

---

## ✅ Model Artifacts

All trained models are saved under the `models/` directory:
- `tuned_lr_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`
- `scaler.pkl`, `pca.pkl`, `label_encoder_name.joblib`, etc.

---

## 📊 Evaluation

- Classification: Accuracy, ROC-AUC, confusion matrix
- Regression: RMSE, R², MAE
- Recommendation: Contextual validation (name/place/price logic)

---

## 🚀 Deployment & Workflow Automation

### Streamlit + Flask Apps:
- UI for all models is embedded inside the notebooks.
- Uses `ngrok` (if needed) to expose endpoints publicly in Colab.
- Predictions are accessible through interactive forms.

### MLflow:
- Tracks experiments for Flight Price Prediction.
- Records metrics, parameters, and model artifacts.

### Apache Airflow:
- A DAG executes `load_data → preprocess → train → predict` pipeline.
- Run via Docker container from `Airflow/` folder.


---


## 🙋 Author

**Ayush Dattatray Bhagat**  
Voyage Analytics Capstone Project – AlmaBetter  
2025
