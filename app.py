import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")

# Load dataset to get columns
df = pd.read_csv(r"C:\Users\fardh\OneDrive\Desktop\numpy\customer-churn-prediction\customer_churn.csv")

# Preprocessing for user input
def preprocess_input(user_input):
    df_temp = df.copy()

    # Drop target + customerID
    df_temp = df_temp.drop(["customerID", "Churn"], axis=1)

    # Append user input
    df_temp = pd.concat([df_temp, user_input], ignore_index=True)

    # Encode categorical columns
    for col in df_temp.select_dtypes(include="object"):
        df_temp[col] = df_temp[col].astype("category").cat.codes

    return df_temp.tail(1)

# Streamlit UI
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 72)
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

# Create input DataFrame
user_input = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone],
    "MultipleLines": [multiple],
    "InternetService": [internet],
    "OnlineSecurity": [online_sec],
    "OnlineBackup": [online_backup],
    "TechSupport": [tech_support],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# Predict button
if st.button("Predict Churn"):
    processed = preprocess_input(user_input)
    prob = model.predict_proba(processed)[0][1]
    pred = model.predict(processed)[0]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")

    if pred == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")