
# 🧠 ML-Based Employee Attrition Prediction & Risk Scoring System

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen)](https://attritionpredictionproject-os52zktbpxbtjruhwisk2u.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue)](https://tableau.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 🔗 **Live Demo:** [Click here to open the app](https://attritionpredictionproject-os52zktbpxbtjruhwisk2u.streamlit.app)

---

## 📌 Project Overview

An end-to-end machine learning project that predicts **which employees are at risk of leaving a company** and provides HR teams with a live, interactive tool to act on those predictions — no coding required.

Built entirely solo — from raw data collection to a deployed production web application.

---

## 🎯 Problem Statement

Employee attrition is costly for organizations. This project:
- Identifies employees most likely to leave using historical HR data
- Scores each employee by attrition risk level
- Provides an easy-to-use live web app for HR teams

---

## 🗂️ Dataset

| Detail | Info |
|---|---|
| Source | IBM HR Analytics Employee Attrition Dataset |
| Records | 1,470 employee records |
| Features | 35 columns (age, salary, overtime, satisfaction, department, etc.) |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Machine learning models |
| Matplotlib & Seaborn | Exploratory data analysis charts |
| Streamlit | Live web application deployment |
| Tableau | Interactive business dashboard |
| GitHub | Version control |

---

## 🚀 End-to-End Pipeline

```
Raw Data → Cleaning → EDA → Feature Engineering → ML Modeling → Evaluation → Deployment
```

### Step 1 — Data Cleaning
- Loaded 1,470 employee records
- Handled missing values and removed duplicates
- Encoded categorical variables and scaled numerical features

### Step 2 — Exploratory Data Analysis (EDA)
- Analyzed attrition patterns across departments, salary bands, and overtime
- Key finding: **Low salary + overtime = highest attrition risk**
- Visualized trends using heatmaps, bar charts, and distribution plots

### Step 3 — Feature Engineering
- Selected top predictors using correlation analysis
- Engineered new features from existing columns
- Applied one-hot encoding and label encoding

### Step 4 — Machine Learning Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 85% | 0.82 |
| Random Forest | 88% | 0.87 |

✅ **Random Forest selected** as final model based on ROC-AUC performance

### Step 5 — Deployment
- Built interactive **Streamlit web app** — HR teams input employee data and get instant risk prediction
- Connected to **Tableau dashboard** for visual analytics and trend monitoring

---

## 📊 Key Business Insights

- 🔴 Employees working **overtime are 3x more likely** to leave
- 🔴 **Low monthly income** is the strongest predictor of attrition
- 🟡 **Poor work-life balance** score strongly linked to higher attrition
- 🟡 **Sales department** shows highest attrition vs other departments

---

## 🖥️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/rahangdalesapna951-arch/AttritionPredictionProject.git

# 2. Navigate to project folder
cd AttritionPredictionProject

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

> **Windows users:** Double-click `run_dashboard.bat` for easy launch

---

## 📁 Project Structure

```
AttritionPredictionProject/
│
├── data/
│   └── HR_Employee_Attrition.csv      # Raw dataset
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb         # Full analysis notebook
│
├── app.py                              # Streamlit web application
├── model.pkl                           # Trained Random Forest model
├── requirements.txt                    # Python dependencies
├── run_dashboard.bat                   # Windows easy launch
└── README.md                           # Project documentation
```

---

## 👩‍💻 About the Author

**Sapna Rahangdale** — Data Scientist | Python · SQL · Power BI · Tableau · ML

| | |
|---|---|
| 🌐 Live App | [Open Streamlit App](https://attritionpredictionproject-os52zktbpxbtjruhwisk2u.streamlit.app) |
| 💼 LinkedIn | [linkedin.com/in/sapna-rahangdale-972732169](https://linkedin.com/in/sapna-rahangdale-972732169) |
| 🐙 GitHub | [github.com/rahangdalesapna951-arch](https://github.com/rahangdalesapna951-arch) |
| 📧 Email | rahangdalesapna951@gmail.com |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
