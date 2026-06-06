# preprocessing_label_encoding_v2.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 🔹 Load the dataset (update the path as needed)
df = pd.read_csv("data/synthetic_loan_fraud.csv")

print("📄 Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print("\nFirst few rows before encoding:")
print(df.head())

# 🔹 Columns in the dataset
columns = [
    'name', 'dob', 'gender', 'email', 'phone', 'address', 'pan', 'aadhaar',
    'loan_type', 'loan_amount', 'income', 'credit_score', 'debt_to_income_ratio',
    'loan_purpose', 'application_channel', 'repayment_status',
    'loan_approval_status', 'ip_address', 'device_id', 'user_agent',
    'city', 'state', 'geo_lat', 'geo_long', 'form_fill_time_seconds',
    'login_time_hour', 'failed_otp_count', 'num_txn_last_30d',
    'shared_phone_count', 'shared_device_count', 'shared_ip_count',
    'email_domain_count', 'aadhaar_verified', 'pan_verified',
    'phone_verified', 'doc_match_score', 'is_fraud'
]

# 🔹 Identify categorical columns (non-numeric)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# You can also explicitly remove unique identifiers that shouldn’t be encoded
ignore_cols = ['name', 'dob', 'email', 'phone', 'aadhaar', 'pan', 'address', 'ip_address', 'device_id', 'user_agent']
categorical_cols = [col for col in categorical_cols if col not in ignore_cols]

# 🔹 Initialize LabelEncoder
le = LabelEncoder()

# 🔹 Encode categorical columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✅ Encoded column: {col}")

# 🔹 Save preprocessed dataset
df.to_csv("data/synthetic_loan_fraud.csv", index=False)

print("\n🎯 Label Encoding Complete!")
print(f"Encoded columns: {categorical_cols}")
print("Saved as: data/synthetic_loan_fraud.csv")
