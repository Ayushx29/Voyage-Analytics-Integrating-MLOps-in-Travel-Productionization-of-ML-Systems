# Voyage-Analytics-Integrating-MLOps-in-Travel-Productionization-of-ML-Systems

This project integrates Machine Learning and (simulated) MLOps practices to build and automate ML pipelines tailored for the travel domain. It consists of three key modules:

- 🧑 Gender Classification based on user profiles
- 🏨 Hotel Recommendation based on budget and place
- 💸 Flight Price Prediction using booking parameters

All models are trained, evaluated, and deployed in **Google Colab**, with model serialization and basic logging to simulate MLOps practices.

---

## 📁 Project Structure

├── Gender_Classification.ipynb
├── Hotel_Prediction.ipynb
├── Flights_Price_Prediction.ipynb
├── models/
│ └── *.pkl / *.joblib
├── users.csv / hotels.csv / flights.csv


---

## 🧠 Modules Overview

### 1. 🧑 Gender Classification
- Inputs: User Code, Company, Name, Age
- Features: PCA on SentenceTransformer embeddings
- Model: Tuned Logistic Regression
- Output: Predicted Gender (Male/Female)

### 2. 🏨 Hotel Recommendation
- Inputs: Name, Budget, Place
- Model: Decision Trees predicting hotel name, place, and price
- Output: Hotel suggestion with estimated price

### 3. 💸 Flight Price Prediction
- Inputs: Airline, Source, Destination, Date, Stops
- Models: Linear, Ridge, Lasso, ElasticNet, XGBoost, Random Forest
- Output: Predicted flight fare in ₹

---

## ✅ Model Artifacts

All trained models are stored in `/models`:
- `tuned_lr_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`, etc.
- `label_encoder_name.joblib`, `scaling.pkl`, etc.

---

## 📊 Evaluation

- Classification: Accuracy, ROC-AUC, confusion matrix
- Regression: RMSE, R², MAE
- Recommendation: Contextual validation (name/place/price logic)

---

## 🚀 Deployment

- Deployed using **Flask** with **ngrok** and also **Streamlit** to allow public API access from Colab
- Frontend HTML forms embedded in notebooks
- Model predictions accessible through user inputs

---

## ⚠️ Note on MLOps Scope and Environment

> Due to the limitations of the Google Colab environment, traditional MLOps tools such as **Apache Airflow**, **Jenkins**, **Docker**, and **MLflow UI** were not used in this version of the project.
>
> However, the core focus on machine learning — including **data preprocessing, model training, evaluation, and saving artifacts** — has been implemented and validated across three key modules.
>
> A **separate version** of this project includes **simulated DAG pipelines** and **experiment logging** using Python-based orchestration and CSV tracking to reflect MLOps principles in a Colab-compatible manner.
>
> This submission prioritizes clarity, model accuracy, and reproducibility over external orchestration due to platform constraints.

---

## 🙋 Author

**Ayush Dattatray Bhagat**  
Voyage Analytics Capstone Project – AlmaBetter  
2025
