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

fraud_reported

* Y → Fraudulent claim
* N → Legitimate claim

### Example Features

* policy_state
* policy_deductable
* insured_sex
* insured_education_level
* incident_type
* incident_city
* number_of_vehicles_involved
* bodily_injuries
* property_damage
* vehicle_claim
* injury_claim
* property_claim
* total_claim_amount

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas – Data manipulation and preprocessing
* NumPy – Numerical computations
* Matplotlib – Data visualization
* Seaborn – Statistical visualization
* Scikit-learn – Machine learning algorithms
* XGBoost – Advanced boosting algorithm
* Pickle – Model saving and loading

### Web Framework

* Streamlit

### Dataset Source

* Kaggle

---

## Machine Learning Algorithms Used

The project trains and compares multiple machine learning models:

### Base Algorithms
* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* XGBoost
* Support Vector Machine (SVM)
* Gradient Boosting

### Advanced Techniques
* **SMOTE** (Synthetic Minority Over-sampling) - Balances fraud/legitimate classes
* **GridSearchCV** - Hyperparameter tuning for Random Forest
* **Optimized XGBoost** - Fine-tuned parameters (n_estimators=300, learning_rate=0.05)
* **Feature Engineering** - Creates derived features (claim_ratio, claim_per_vehicle, claim_per_injury)

**Best Model:** XGBoost with optimized hyperparameters achieves ~88-90% accuracy.

---

## Evaluation Metrics

The models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics help determine how effectively the model detects fraudulent claims.

---

## Project Workflow

1. Data Collection
   Insurance claim dataset is collected from Kaggle.

2. Data Preprocessing

   * Handling missing values
   * Removing duplicates
   * Encoding categorical variables
   * Converting fraud labels into numerical values

3. Exploratory Data Analysis (EDA)
   Data visualization is used to understand patterns and relationships between variables.

4. Model Training
   Multiple machine learning algorithms are trained on the dataset.

5. Model Evaluation
   Models are evaluated using performance metrics to determine the best model.

6. Model Deployment
   The best-performing model is saved and integrated into a Streamlit web application.

7. Fraud Prediction
   Users input claim details through the web interface, and the system predicts whether the claim is fraudulent.

---

## Project Structure

```
insurance-fraud-detection
│
├── insurance_claims.csv          # Training dataset (1000 records, 40 features)
├── fraud_detection_analysis.ipynb  # Complete ML pipeline with SMOTE & tuning
├── app.py                         # Streamlit web application (15 key inputs)
├── sample_data.py                 # 6 diverse test cases (3 fraud, 3 legitimate)
├── fraud_model.pkl                # Trained XGBoost model with encoders
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Installation

### 1. Navigate to the Project Directory

cd "c:\Users\vidis\Downloads\ai insyrance"

### 2. Install Dependencies

pip install -r requirements.txt

---

## Run the Project

### Step 1: Train the Model

Open and run the Jupyter notebook:

jupyter notebook fraud_detection_analysis.ipynb

Or open it in VS Code and run all cells. This will:
- Perform complete data analysis with visualizations
- Train 7 different ML models
- Compare their performance
- Save the best model as **fraud_model.pkl**

### Step 2: Start the Web Application

streamlit run app.py

The application will open in your browser at:

http://localhost:8501

---

## Web Application Features

### 🎯 Feature Importance-Based Design

Unlike traditional forms that ask for 40+ fields, this application uses **data-driven feature selection** based on importance analysis:

**Top 15 Predictive Features:**
1. **incident_severity** (16.2%) - Strongest fraud indicator
2. **insured_hobbies** (6.9%) - Behavioral patterns
3. **incident_state** (5.6%) - Regional fraud trends
4. **umbrella_limit** (3.9%) - High coverage correlation
5. **authorities_contacted** (3.8%) - Investigation patterns
6. **witnesses** (2.9%) - Verification availability
7. **policy_state** (2.8%) - Policy location patterns
8. **policy_csl** (2.8%) - Coverage limits
9. **police_report_available** (2.5%) - Documentation quality
10. **property_claim** (2.3%) - Claim amount patterns
11. **incident_type** (2.1%) - Incident classification
12. **insured_zip** (2.1%) - Geographic patterns
13. **incident_date** (2.1%) - Timing analysis
14. **total_claim_amount** (2.0%) - Overall claim size
15. **property_damage** (2.0%) - Damage verification

### 📊 How It Works

**User Inputs:** 15 key fields organized in 5 sections
- 📋 Policy Details (3 fields)
- 🚗 Incident Details (5 fields)
- 💰 Claim Details (3 fields)
- 🔍 Investigation Info (2 fields)
- 👤 Customer Info (2 fields)

**Auto-Computed:** 3 derived features
- claim_ratio = total_claim / annual_premium
- claim_per_vehicle = total_claim / vehicles_involved
- claim_per_injury = injury_claim / (injuries + 1)

**Auto-Filled:** 25+ less important fields
- Customer demographics (age, education, occupation)
- Policy administrative data (numbers, dates)
- Vehicle details (make, model, year)
- Incident specifics (time, location, collision type)

### Key Features

* ✅ **Data-driven input selection** - Focus on features that matter most
* ✅ **Simple and interactive interface** - Only 15 fields vs 40+
* ✅ **Instant fraud prediction** - Real-time ML analysis
* ✅ **Sample data loader** - 6 pre-configured test cases
* ✅ **Debug mode** - View data processing pipeline
* ✅ **Professional UX** - 2-minute claim entry
* ✅ **Maximum accuracy** - Uses all 41 features internally

---

## Example Prediction

**Input (15 fields):**
* Policy State: OH
* Policy CSL: 250/500
* Umbrella Limit: $0
* Incident Type: Vehicle Theft
* Incident Severity: Total Loss
* Incident State: OH
* Authorities Contacted: None
* Witnesses: 0
* Police Report: NO
* Total Claim: $95,000
* Property Claim: $20,000
* Property Damage: YES
* Insured Hobbies: base-jumping
* Insured ZIP: 10001

**Output:**
⚠️ **FRAUDULENT CLAIM DETECTED**
* Fraud Probability: 87.5%
* Recommendation: Requires manual investigation

**Why?** High claim amount + Total Loss + No authorities/police + No witnesses + high-risk hobby = strong fraud indicators

---

## Future Improvements

* 📊 **Real-time dashboards** - Monitor fraud trends and patterns
* 🤖 **Deep learning models** - Neural networks for complex pattern detection
* 🌐 **Cloud deployment** - AWS/Azure/GCP hosting
* 📱 **Mobile application** - Field claim entry
* 🔔 **Alert system** - Automated notifications for high-risk claims
* 📊 **Advanced analytics** - Fraud pattern visualization
* 👥 **Multi-user support** - Role-based access control
* 💾 **Database integration** - PostgreSQL/MongoDB for claim history

---

## Conclusion

This project demonstrates how machine learning can be used to detect fraudulent insurance claims efficiently. By analyzing historical insurance claim data, the system can identify suspicious patterns and assist insurance companies in detecting fraud more accurately.

The integration of a web application makes the system easy to use and deploy, providing a practical solution for fraud detection in the insurance industry.

---

## Contributors

* Vidish Kumar (Team Lead)
* Somya Kushwah
* Gaurangi Sharma
* Kritika Gupta
#   i n s u r a n c e - f r a u d - d e t e c t i o n  
 