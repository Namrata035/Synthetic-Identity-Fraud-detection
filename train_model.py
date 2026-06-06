# train_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Paths
DATA_PATH = "data/synthetic_loan_fraud.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# Preprocessing
# Fill missing credit_score with median
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())

# Select only features that are available at inference (avoid leakage and PII)
selected_features = [
    'gender',
    'loan_type',
    'loan_amount',
    'income',
    'credit_score',
    'debt_to_income_ratio',
    'loan_purpose',
    'application_channel',
    'repayment_status',
    # 'loan_approval_status' is intentionally excluded to prevent target leakage
    'is_fraud'
]
df_model = df[selected_features].copy()

# Identify categorical columns explicitly (string-like in CSV)
cat_cols = ['gender', 'loan_type', 'loan_purpose', 'application_channel', 'repayment_status']
print("Categorical cols:", cat_cols)

# Encode categorical columns with explicit mapping dicts including an UNK token
label_encoders = {}
for col in cat_cols:
    unique_vals = sorted(df_model[col].astype(str).unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    mapping['__UNK__'] = len(mapping)
    df_model[col] = df_model[col].astype(str).map(lambda v: mapping.get(v, mapping['__UNK__']))
    label_encoders[col] = mapping

# Features and target
X = df_model.drop('is_fraud', axis=1).values
y = df_model['is_fraud'].values

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler & encoders & columns order
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(df_model.columns.tolist(), os.path.join(MODEL_DIR, "model_columns.pkl"))

# Reshape for CNN+LSTM: (samples, timesteps, features) => use timesteps=1
X_reshaped = np.expand_dims(X_scaled, axis=1)  # shape (n, 1, features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# Build model
n_timesteps = X_train.shape[1]   # 1
n_features = X_train.shape[2]    # num features

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Evaluate
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Save model
model.save(os.path.join(MODEL_DIR, "cnn_lstm_fraud_model.h5"))
print("[✓] Saved model to", os.path.join(MODEL_DIR, "cnn_lstm_fraud_model.h5"))
