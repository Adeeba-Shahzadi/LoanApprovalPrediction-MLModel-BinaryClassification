# Student Loan Approval Prediction

This project predicts whether a **loan application will be approved or rejected** based on applicant details. It uses **Machine Learning classification models** like Logistic Regression and Decision Tree.

---

## 📌 Project Overview
Loan approval prediction is a **binary classification problem**:
- **Approved (1)**
- **Rejected (0)**  

It use ML models to train on past loan application data and predict outcomes for new applicants.

---
## 📂 Dataset

**Source:** Kaggle - Loan Approval Prediction Dataset  

**Number of Rows:** 4269  
**Number of Columns:** 13  

**Features:**
- `loan_id` → Unique ID of the loan application  
- `no_of_dependents` → Number of dependents of applicant  
- `education` → Graduate / Not Graduate  
- `self_employed` → Yes / No  
- `income_annum` → Annual income of the applicant  
- `loan_amount` → Loan amount requested  
- `loan_term` → Loan repayment term in months  
- `cibil_score` → Credit score of the applicant  
- `residential_assets_value` → Value of residential assets  
- `commercial_assets_value` → Value of commercial assets  
- `luxury_assets_value` → Value of luxury assets  
- `bank_asset_value` → Value of bank assets  
- `loan_status` → Target variable (Approved=1, Rejected=0)  

**Preprocessing Steps:**
- Removed duplicates  
- Handled missing values  
- Dropped `loan_id` (irrelevant for prediction)  
- Encoded categorical variables into numeric form  

---

## ⚙️ Models Used
1. **Logistic Regression**
   - Best for binary classification.
   - Outputs probabilities between 0 and 1.
   - Helps in understanding the relationship between features and target.

2. **Decision Tree Classifier**
   - Splits data into branches using conditions.
   - Easy to visualize and interpret.
   - Can overfit if not tuned properly.

---

## 📊 Evaluation Metrics
We used the following metrics to evaluate model performance:

- **Accuracy** → Correct predictions / Total predictions  
- **Precision** → Out of predicted "Approved", how many were actually "Approved"  
- **Recall (Sensitivity)** → Out of actual "Approved", how many were correctly predicted  
- **F1 Score** → Harmonic mean of Precision & Recall  
- **Confusion Matrix** → Table showing True/False Positives and Negatives  
---

## 🛠️ Installation

1. Clone the repository (or download the files):
```bash
   git clone https://github.com/Adeeba-Shahzadi/LoanApprovalPrediction-BinaryClassification.git
   cd LoanApprovalPrediction-BinaryClassification
```

2. Install required dependencies:
```bash
    pip install -r requirements.txt
```

## 📂 Files
- LoanApprovalPrediction.ipynb → Jupyter Notebook with step-by-step implementation.

- loan_approval_dataset.csv → Dataset for training
- loan_prediction.py → Python script version.

- requirements.txt → Required libraries.

- README.md → Project documentation.

## ▶️ How to Run
- Run with Jupyter Notebook:
```bash
    Copy code
    jupyter notebook loan_prediction.ipynb
```
- Run with Python:
```bash
    Copy code
    python loan_prediction.py
```
