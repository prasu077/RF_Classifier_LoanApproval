import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
with open("bagging_loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Predict whether a loan will be **Approved or Rejected**")

# ---------------- INPUTS ----------------
no_of_dependents = st.number_input("Number of Dependents", 0, 10, 1)

education = st.selectbox(
    "Education",
    options=[0, 1],
    format_func=lambda x: "Graduate" if x == 0 else "Not Graduate"
)

self_employed = st.selectbox(
    "Self Employed",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    max_value=50000000,
    value=300000,
    step=10000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    max_value=50000000,
    value=500000,
    step=50000
)

loan_term = st.number_input("Loan Term (Years)", 1, 30, 10)
cibil_score = st.number_input("CIBIL Score", 300, 900, 650)

asset_level = st.selectbox(
    "Overall Asset Strength",
    ["Low", "Medium", "High"]
)

# ---------------- ASSET MAPPING ----------------
if asset_level == "Low":
    res_assets, com_assets, lux_assets, bank_assets = 500000, 200000, 200000, 100000
elif asset_level == "Medium":
    res_assets, com_assets, lux_assets, bank_assets = 3000000, 1500000, 2000000, 1000000
else:
    res_assets, com_assets, lux_assets, bank_assets = 10000000, 8000000, 6000000, 4000000


# ---------------- PREDICTION ----------------
if st.button("Predict Loan Status"):
    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        res_assets,
        com_assets,
        lux_assets,
        bank_assets
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

if prediction[0] == 0:
    st.success("✅ Loan Approved")
else:
    st.error("❌ Loan Rejected")

approved_prob = model.predict_proba(input_data)[0][0] * 100
st.info(f"Approval Probability: {round(approved_prob, 2)}%")
