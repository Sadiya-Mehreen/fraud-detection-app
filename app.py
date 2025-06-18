# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# Page title
st.title("💳 Online Payment Fraud Detection")

st.markdown("Enter transaction details to check for possible fraud:")

# Input fields
step = st.number_input("Step (Hour)", min_value=1)
txn_type = st.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'CASH_IN'])
amount = st.number_input("Transaction Amount")
oldbalanceOrg = st.number_input("Old Balance (Sender)")
newbalanceOrig = st.number_input("New Balance (Sender)")

# Encode 'type'
type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
type_encoded = type_mapping[txn_type]

# Predict button
if st.button("Predict Fraud"):
    # Create input dataframe
    input_df = pd.DataFrame([[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig]],
                            columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig'])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # Output result
    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted to be FRAUDULENT with {proba*100:.2f}% probability.")
    else:
        st.success(f"✅ This transaction is NOT fraudulent. Probability of fraud: {proba*100:.2f}%")
