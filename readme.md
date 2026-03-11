# Insurance Fraud Detection Using Machine Learning

## Project Overview

Insurance fraud is a major problem faced by insurance companies worldwide. Fraudulent claims lead to significant financial losses and increase the cost of insurance premiums for honest customers. This project aims to develop a **Machine Learning-based system that predicts whether an insurance claim is fraudulent or legitimate** using historical insurance claim data.

The system analyzes several parameters such as **policy details, accident information, customer attributes, and claim amounts** to identify suspicious claims. A trained machine learning model processes the claim details and predicts whether the claim is **fraudulent or genuine**.

The final system is deployed as a **web application using Streamlit**, allowing users to input insurance claim information and receive fraud predictions instantly.

---

## Problem Statement

Insurance companies process thousands of claims every day. Among these claims, some may be fraudulent. Detecting these fraudulent claims manually is time-consuming and inefficient. Therefore, an automated system is required to detect fraudulent claims using machine learning techniques.

The objective of this project is to build a machine learning model that can analyze insurance claim data and predict whether a claim is fraudulent or legitimate.

---

## Dataset

The dataset used in this project is obtained from Kaggle.

Dataset Name: **Insurance Fraud Detection Dataset**

The dataset contains multiple attributes related to insurance policies, accidents, and claim amounts. These features are used to train the machine learning model.

### Target Variable
# Insurance Fraud Detection Using Machine Learning

## Overview
This project predicts whether an insurance claim is fraudulent using machine learning and deploys the model with Streamlit.

The solution includes:
1. End-to-end notebook workflow: EDA, preprocessing, feature engineering, model training, tuning, evaluation.
2. Trained model artifact (`fraud_model.pkl`) with preprocessing objects.
3. Streamlit web app for quick fraud risk assessment.

## Problem Statement
Insurance fraud creates major losses for insurers and increases premium costs for honest customers. Manual claim screening is slow and inconsistent. This project builds an automated fraud screening system to support faster and better claim decisions.

## Dataset
- Source: Kaggle insurance claim fraud dataset
- Target column: `fraud_reported`
- Label mapping:
1. `Y` = Fraudulent claim
2. `N` = Legitimate claim

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- Streamlit

## ML Pipeline
The notebook `fraud_detection_analysis.ipynb` includes:
1. Data cleaning and missing value handling
2. Categorical encoding with `LabelEncoder`
3. Feature engineering, including:
   - `claim_ratio`
   - `claim_per_vehicle`
   - `claim_per_injury`
   - `coverage_ratio`
   - `injury_severity_score`
   - `night_incident`
   - `incident_month`, `incident_day`, `incident_weekday`
4. Feature selection with `SelectKBest(f_classif, k=25)`
5. Feature scaling with `StandardScaler`
6. Model training and comparison
7. Tuned XGBoost with class weighting (`scale_pos_weight`)
8. Evaluation with Accuracy, Precision, Recall, F1, ROC AUC, ROC Curve
9. Cross-validation (5-fold F1)

## Current App Behavior
The Streamlit app accepts a compact set of high-value inputs and returns:
1. Fraud probability
2. Risk level:
   - `Low`
   - `Medium`
   - `High`
3. Visual prediction cards and probability breakdown

## Project Structure
```text
insurance-fraud-detection/
├── insurance_claims.csv
├── fraud_detection_analysis.ipynb
├── app.py
├── sample_data.py
├── fraud_model.pkl
├── requirements.txt
└── readme.md
```

## Setup
1. Open terminal in project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
1. Train and export model from notebook:

```bash
jupyter notebook fraud_detection_analysis.ipynb
```

Run all cells to regenerate `fraud_model.pkl`.

2. Start Streamlit app:

```bash
streamlit run app.py
```

## Notes
- If notebook preprocessing changes, regenerate `fraud_model.pkl` before running the app.
- The app supports model artifacts that include optional selector (`SelectKBest`) and uses it when available.

## Contributors
- Vidish Kumar (Team Lead)
- Somya Kushwah
- Gaurangi Sharma
- Kritika Gupta

