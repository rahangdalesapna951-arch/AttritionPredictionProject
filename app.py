import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── PAGE SETUP ─────────────────────────────────────────
st.set_page_config(
    page_title = "Attrition Risk Dashboard",

       layout     = "wide"
)

# ── LOAD MODEL ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── LOAD DATA ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Palo Alto Networks.csv')
    return df

df = load_data()

# ── PREPROCESS ─────────────────────────────────────────
def preprocess(df):
    df2 = df.copy()
    le  = LabelEncoder()
    df2['Gender']   = le.fit_transform(df2['Gender'])
    df2['OverTime'] = le.fit_transform(df2['OverTime'])
    df2 = pd.get_dummies(df2, columns=[
        'BusinessTravel', 'Department', 'EducationField',
        'JobRole', 'MaritalStatus'
    ])
    if 'Attrition' in df2.columns:
        df2 = df2.drop('Attrition', axis=1)
    return df2

# ── PREDICT RISK ───────────────────────────────────────
def get_risk(prob):
    if prob < 0.30:   return 'Low Risk',    '#639922'
    elif prob < 0.60: return 'Medium Risk', '#EF9F27'
    else:             return 'High Risk',   '#E24B4A'

# ── SIDEBAR ────────────────────────────────────────────
st.sidebar.image("logo1.JPG", width=200)

st.sidebar.title("HR Attrition Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Risk Overview",
    "Employee Profiles",
    "What-If Explorer"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
dept_filter = st.sidebar.selectbox(
    "Department",
    ["All Departments"] + df['Department'].unique().tolist()
)
risk_threshold = st.sidebar.slider(
    "High Risk Threshold (%)", 0, 100, 60
)

# ── PREPARE PREDICTIONS ────────────────────────────────
df_processed = preprocess(df)
df_processed_scaled = scaler.transform(df_processed)
probabilities = model.predict_proba(df_processed_scaled)[:,1]

df['Attrition_Probability'] = probabilities
df['Risk_Category'] = [get_risk(p)[0] for p in probabilities]

# Apply department filter
if dept_filter != "All Departments":
    df_filtered = df[df['Department'] == dept_filter]
else:
    df_filtered = df

# ══════════════════════════════════════════════════════
# PAGE 1 — RISK OVERVIEW
# ══════════════════════════════════════════════════════
if page == "Risk Overview":

    st.title("Attrition Risk Overview")
    st.markdown("---")

    # Metric cards
    total     = len(df_filtered)
    high_risk = len(df_filtered[df_filtered['Risk_Category'] == 'High Risk'])
    med_risk  = len(df_filtered[df_filtered['Risk_Category'] == 'Medium Risk'])
    low_risk  = len(df_filtered[df_filtered['Risk_Category'] == 'Low Risk'])
    avg_prob  = df_filtered['Attrition_Probability'].mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Employees", total)
    col2.metric("High Risk",       high_risk,  delta=f"{high_risk/total*100:.1f}%")
    col3.metric("Medium Risk",     med_risk,   delta=f"{med_risk/total*100:.1f}%")
    col4.metric("Low Risk",        low_risk,   delta=f"{low_risk/total*100:.1f}%")
    col5.metric("Avg Risk Score",  f"{avg_prob:.1f}%")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        risk_counts = df_filtered['Risk_Category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(risk_counts.values,
               labels=risk_counts.index,
               colors=['#639922', '#EF9F27', '#E24B4A'],
               autopct='%1.1f%%',
               startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=2))
        ax.set_title('Employee Risk Categories')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Attrition Rate by Department")
        dept_risk = df_filtered.groupby('Department')['Attrition_Probability'].mean().mul(100).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#639922' if v < 16 else '#EF9F27' if v < 20 else '#E24B4A'
                  for v in dept_risk.values]
        bars = ax.barh(dept_risk.index, dept_risk.values,
                       color=colors, edgecolor='white')
        ax.set_xlabel('Avg Attrition Probability (%)')
        for b, v in zip(bars, dept_risk.values):
            ax.text(v + 0.3, b.get_y() + b.get_height()/2,
                    f'{v:.1f}%', va='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Attrition Rate by Job Role")
    role_risk = df_filtered.groupby('JobRole')['Attrition_Probability'].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#639922' if v < 10 else '#EF9F27' if v < 20 else '#E24B4A'
              for v in role_risk.values]
    bars = ax.barh(role_risk.index, role_risk.values,
                   color=colors, edgecolor='white')
    ax.axvline(16.1, color='gray', linestyle='--', linewidth=1.5, label='Avg 16.1%')
    ax.set_xlabel('Avg Attrition Probability (%)')
    ax.legend()
    for b, v in zip(bars, role_risk.values):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2,
                f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE PROFILES
# ══════════════════════════════════════════════════════
elif page == "Employee Profiles":

    st.title("Employee Risk Profiles")
    st.markdown("---")

    # High risk table
    st.subheader("High Risk Employees — Sorted by Attrition Probability")

    high_risk_df = df_filtered[
        df_filtered['Attrition_Probability'] >= risk_threshold/100
    ][['Department', 'JobRole', 'MonthlyIncome',
       'OverTime', 'Attrition_Probability', 'Risk_Category']].copy()

    high_risk_df['Attrition_Probability'] = (
        high_risk_df['Attrition_Probability'] * 100
    ).round(1).astype(str) + '%'

    high_risk_df = high_risk_df.sort_values(
        'Attrition_Probability', ascending=False
    )
    high_risk_df.columns = [
        'Department', 'Job Role', 'Monthly Income',
        'Overtime', 'Risk Probability', 'Risk Category'
    ]

    st.dataframe(high_risk_df, use_container_width=True)
    st.markdown(f"**{len(high_risk_df)} employees** above {risk_threshold}% risk threshold")

    st.markdown("---")

    # Individual employee lookup
    st.subheader("Look Up Individual Employee")
    emp_index = st.number_input(
        "Enter Employee Index (0 to 1469)",
        min_value=0, max_value=1469, value=0
    )

    emp = df.iloc[emp_index]
    prob = emp['Attrition_Probability']
    risk, color = get_risk(prob)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Employee #{emp_index}")
        st.metric("Department",    emp['Department'])
        st.metric("Job Role",      emp['JobRole'])
        st.metric("Monthly Income",f"Rs {emp['MonthlyIncome']:,}")
        st.metric("Age",           emp['Age'])
        st.metric("Overtime",      emp['OverTime'])
        st.metric("Marital Status",emp['MaritalStatus'])

    with col2:
        st.markdown("### Risk Assessment")
        st.metric("Attrition Probability", f"{prob*100:.1f}%")
        if risk == 'High Risk':
            st.error(f"Risk Category: {risk}")
        elif risk == 'Medium Risk':
            st.warning(f"Risk Category: {risk}")
        else:
            st.success(f"Risk Category: {risk}")

        st.markdown("**Key Factors:**")
        if emp['OverTime'] == 'Yes':
            st.write("Working overtime — high risk factor!")
        if emp['MonthlyIncome'] < 4000:
            st.write("Low salary — high risk factor!")
        if emp['MaritalStatus'] == 'Single':
            st.write("Single — medium risk factor!")
        if emp['JobSatisfaction'] <= 2:
            st.write("Low job satisfaction — risk factor!")

# ══════════════════════════════════════════════════════
# PAGE 3 — WHAT-IF EXPLORER
# ══════════════════════════════════════════════════════
elif page == "What-If Explorer":

    st.title("What-If Explorer")
    st.markdown("Change employee details below and see how risk changes!")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adjust Employee Factors")
        monthly_income  = st.slider("Monthly Income (Rs)", 1000, 20000, 4000, 500)
        job_satisfaction= st.slider("Job Satisfaction",    1, 4, 2)
        overtime        = st.selectbox("Overtime",         ["No", "Yes"])
        work_life       = st.slider("Work-Life Balance",   1, 4, 2)
        stock_option    = st.slider("Stock Option Level",  0, 3, 0)
        distance        = st.slider("Distance from Home",  1, 29, 10)
        age             = st.slider("Age",                 18, 60, 30)
        years_company   = st.slider("Years at Company",    0, 40, 2)

    with col2:
        st.subheader("Predicted Risk")

        # Simple risk calculation based on key factors
        base_prob = 0.30

        if overtime == "Yes":      base_prob += 0.25
        if monthly_income < 3000:  base_prob += 0.20
        elif monthly_income < 5000:base_prob += 0.10
        if job_satisfaction == 1:  base_prob += 0.10
        elif job_satisfaction == 2:base_prob += 0.05
        if work_life == 1:         base_prob += 0.08
        if stock_option == 0:      base_prob += 0.08
        if distance > 20:          base_prob += 0.05
        if age < 25:               base_prob += 0.08
        if years_company < 2:      base_prob += 0.08

        base_prob = min(base_prob, 0.99)
        base_prob = max(base_prob, 0.01)

        risk, color = get_risk(base_prob)

        st.markdown(f"### Attrition Probability")
        st.markdown(f"# {base_prob*100:.0f}%")

        if risk == 'High Risk':
            st.error(f"Risk Category: {risk}")
            st.markdown("**HR Action Required:**")
            st.write("Reduce overtime immediately")
            st.write("Review and increase salary")
            st.write("Schedule retention meeting")
        elif risk == 'Medium Risk':
            st.warning(f"Risk Category: {risk}")
            st.markdown("**HR Recommendation:**")
            st.write("Monitor employee closely")
            st.write("Improve work-life balance")
            st.write("Consider salary review")
        else:
            st.success(f"Risk Category: {risk}")
            st.markdown("**Employee is stable!**")
            st.write("Continue current support")
            st.write("Regular check-ins recommended")

        st.markdown("---")
        st.markdown("**Try this:** Set Overtime=Yes and Income=2000")
        st.markdown("Then set Overtime=No and Income=8000")
        st.markdown("Watch how the risk changes!")