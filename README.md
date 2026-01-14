# 🧠 Alzheimer’s Disease Prediction using Machine Learning

This project focuses on predicting Alzheimer’s disease using patient demographic, lifestyle, cognitive, and behavioral data.  
The dataset is **imbalanced**, so special care is taken in preprocessing, model training, and evaluation using medically appropriate metrics.

---

## 📌 Problem Statement

Early detection of Alzheimer’s disease is critical for timely intervention.  
The goal of this project is to build and compare multiple machine learning models to accurately predict disease diagnosis while handling **class imbalance**.

---

## 📊 Dataset Description

The dataset contains patient-level information including:

- Demographics: Age, Gender, Ethnicity, Education Level
- Lifestyle factors: BMI, Smoking, Alcohol Consumption, Physical Activity, Diet Quality
- Cognitive & behavioral symptoms: Memory Complaints, Confusion, Disorientation, Forgetfulness, etc.
- Functional ability: Activities of Daily Living (ADL)
- Target variable: **Diagnosis** (0 = No Disease, 1 = Disease)

### ⚠️ Class Imbalance
The dataset is imbalanced, with fewer positive (disease) cases.  
Therefore, **Accuracy alone is not a reliable metric**.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removed non-informative columns:
  - `PatientID`
  - `DoctorInCharge`
- Train–test split with stratification
- Feature scaling using `StandardScaler`
- Pipelines created using `make_pipeline`

---

### 2️⃣ Models Implemented

The following models were trained and tuned using **GridSearchCV**:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes

For imbalance handling:
- `class_weight='balanced'` used where applicable
- Stratified cross-validation

---

### 3️⃣ Model Evaluation Metrics

Due to the medical and imbalanced nature of the problem, models were evaluated using:

- Precision
- Recall (most important)
- F1-score
- ROC-AUC

---

## 🏆 Results Summary

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Random Forest | 0.939 | 0.938 | 0.888 | **0.912** | **0.945** |
| Decision Tree | 0.923 | 0.884 | **0.901** | 0.893 | 0.936 |
| Logistic Regression | 0.814 | 0.689 | 0.862 | 0.766 | 0.883 |
| SVM | 0.795 | 0.663 | 0.855 | 0.747 | 0.884 |
| Naive Bayes | 0.772 | 0.671 | 0.697 | 0.684 | 0.850 |
| KNN | 0.707 | 0.633 | 0.408 | 0.496 | 0.738 |

---

## ✅ Final Model Selection

- **Best Overall Model:** 🏆 **Random Forest**
  - Highest F1-score and ROC-AUC
  - Balanced precision and recall

- **Best Recall Model:** 🩺 **Decision Tree**
  - Suitable when minimizing false negatives is critical

---

## 📈 ROC Curve Analysis

ROC curves were plotted for all models to compare class separability.  
Random Forest and Decision Tree showed the strongest discriminative performance.

---

## 🧠 Key Takeaways

- Accuracy is misleading for imbalanced medical datasets
- Recall and F1-score are more reliable evaluation metrics
- Ensemble models perform better in complex healthcare data
- Proper pipelines prevent data leakage

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## 🚀 Future Work

- Precision–Recall curve analysis
- Threshold tuning for higher recall
- Feature importance analysis
- Streamlit-based deployment
- SMOTE-based resampling comparison

---

## 📂 Project Structure
├── alzheimer_disease_prediction_with_imbalanced_classification.ipynb
├── README.md
├── requirements.txt
└── data/
└── dataset.csv
