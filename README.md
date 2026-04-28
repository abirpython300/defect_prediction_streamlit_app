# defect_prediction_streamlit_app
This project is a machine learning-based web application that predicts whether a product will be defective based on production parameters.
# 🔍 Defect Prediction System (Streamlit Web App)

An end-to-end machine learning project that predicts whether a product will be defective based on production parameters. The model is deployed as an interactive **Streamlit web application** for real-time prediction and decision support.

---

# 🚀 Project Overview

In manufacturing, defects lead to increased costs, rework, and inefficiencies.  
This project aims to **predict defects in advance** so that corrective actions can be taken proactively.

👉 The solution combines:
- Data preprocessing & feature selection  
- Logistic regression modeling  
- Model evaluation & validation  
- Deployment using Streamlit  

---

# 🎯 Business Problem

Manufacturing units often face:
- High scrap and rework costs  
- Unpredictable defect occurrences  
- Lack of data-driven decision-making  

---

# 🎯 Objective

To develop a predictive system that:
- Identifies defective products in advance  
- Improves production quality  
- Reduces operational cost  

---

# 📊 Data Sources

The dataset is a consolidated version of multiple industrial systems:

- ERP (Enterprise Resource Planning)  
- MES (Manufacturing Execution Systems)  
- QMS (Quality Management Systems)  
- SCADA (Supervisory Control and Data Acquisition)  
- Machine Master Data  
- Operator Data  

👉 These sources provide:
- Operational metrics  
- Machine conditions  
- Environmental parameters  
- Operator performance  

---

# 🧹 Data Preparation

## ✔ Missing Values
- Only **1.48% missing values**
- Removed to maintain data quality  

## ✔ Outlier Detection
- Used **Boxplot (IQR method)**  
- Identified extreme values in numerical features  

## ✔ Feature Selection
- **ANOVA Test** → Numerical variables  
- **Chi-Square Test** → Categorical variables  

---

# ⚙️ Model Development

- Algorithm: **Logistic Regression**
- Train-Test Split: **80:20**
- Stratified sampling used  

---

# 📈 Model Performance

| Metric | Value |
|------|------|
| Accuracy | 96% |
| Recall (Defect Detection) | 91% |
| ROC-AUC | ~0.99 |

👉 Interpretation:
- High accuracy → overall correctness  
- High recall → strong defect detection  
- ROC → excellent class separation  

---

# 📊 Confusion Matrix Interpretation

- ✅ True Positives → Correct defect detection  
- ❌ False Negatives → Missed defects (minimized)  
- ⚠️ False Positives → manageable  

---

# 🔍 Key Insights

- Scrap percentage is the strongest predictor  
- Rework time significantly increases defect probability  
- Operator training reduces defects  
- Environmental factors have moderate impact  

---

# 🌐 Streamlit Web Application

The model is deployed as an interactive web app.

## Features:
- User input for production parameters  
- Real-time prediction  
- Probability of defect occurrence  
- Simple and user-friendly interface  

---

# 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  

---

# ▶️ How to Run Locally

## Step 1: Clone the repository

```bash
git clone https://github.com/your-username/defect-prediction-streamlit-app.git
cd defect-prediction-streamlit-app
