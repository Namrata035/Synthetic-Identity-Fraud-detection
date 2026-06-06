from flask import Flask, render_template, request, redirect, url_for, session, abort
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import re

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Resolve paths relative to this file to avoid working directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Lazy-loaded artifacts (loaded on first use)
model = None
scaler = None
label_encoders = None
model_columns = None

def get_artifacts():
    global model, scaler, label_encoders, model_columns
    if all(a is not None for a in (model, scaler, label_encoders, model_columns)):
        return model, scaler, label_encoders, model_columns
    try:
        model_path = os.path.join(MODEL_DIR, "cnn_lstm_fraud_model.h5")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
        cols_path = os.path.join(MODEL_DIR, "model_columns.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(enc_path) and os.path.exists(cols_path)):
            abort(500, description="Model artifacts missing. Please run train_model.py.")
        loaded_model = load_model(model_path)
        loaded_scaler = joblib.load(scaler_path)
        loaded_label_encoders = joblib.load(enc_path)
        loaded_model_columns = joblib.load(cols_path)
        model = loaded_model
        scaler = loaded_scaler
        label_encoders = loaded_label_encoders
        model_columns = loaded_model_columns
        return model, scaler, label_encoders, model_columns
    except Exception as exc:
        abort(500, description=f"Failed to load artifacts: {exc}")


# ---------------------------------------------
# ROUTES
# ---------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/loan_page", methods=["POST"])
def loan_page():
    """Step 1 → Step 2: Collect and validate personal info"""
    required_fields = ["name", "dob", "gender", "address", "email", "phone", "pan", "aadhaar"]
    missing = [f for f in required_fields if not (request.form.get(f) or "").strip()]
    if missing:
        abort(400, description=f"Missing required fields: {', '.join(missing)}")

    # ✅ PAN validation — 5 letters followed by 5 digits (example: ABCDE12345)
    pan = request.form.get("pan", "").strip().upper()
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{5}$")
    if not pan_pattern.match(pan):
        abort(400, description="Invalid PAN format. Must be 5 letters followed by 5 digits (e.g., ABCDE12345)")

    # store validated data
    form_data = session.get("form", {})
    for f in required_fields:
        form_data[f] = request.form.get(f).strip()
    form_data["pan"] = pan  # ensure uppercase
    session["form"] = form_data

    return render_template("loan.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ensure artifacts are loaded
    mdl, scl, encs, cols = get_artifacts()

    # collect loan details from page 2
    form_data = session.get("form", {})
    form_data["loan_type"] = (request.form.get("loan_type", "") or "").strip()
    form_data["loan_purpose"] = (request.form.get("loan_purpose", "") or "").strip()
    form_data["application_channel"] = (request.form.get("application_channel", "") or "").strip()
    form_data["repayment_status"] = (request.form.get("repayment_status", "") or "").strip()
    cat_missing = [k for k in ["loan_type", "loan_purpose", "application_channel", "repayment_status"] if not form_data[k]]
    if cat_missing:
        abort(400, description=f"Missing required fields: {', '.join(cat_missing)}")

    # numeric parsing with validation
    def parse_float(name):
        raw = request.form.get(name, "").strip()
        try:
            return float(raw)
        except Exception:
            abort(400, description=f"Invalid value for {name}.")

    loan_amount = parse_float("loan_amount")
    income = parse_float("income")
    credit_score = parse_float("credit_score")
    if income <= 0 or loan_amount < 0 or not (300 <= credit_score <= 900):
        abort(400, description="Invalid numeric inputs. Ensure income>0, loan_amount>=0, 300<=credit_score<=900.")

    # compute DTI server-side
    debt_to_income_ratio = round(loan_amount / income, 2)

    form_data["loan_amount"] = loan_amount
    form_data["income"] = income
    form_data["credit_score"] = credit_score
    form_data["debt_to_income_ratio"] = debt_to_income_ratio
    session["form"] = form_data

    # Build DataFrame with same columns order used in training
    feature_cols = [c for c in cols if c != "is_fraud"]
    row = {c: 0 for c in feature_cols}
    for k, v in form_data.items():
        if k in feature_cols:
            row[k] = v

    df = pd.DataFrame([row], columns=feature_cols)

    # Encode categorical columns
    for col, mapping in encs.items():
        if col in df.columns:
            unk_val = mapping.get("__UNK__", 0)
            df[col] = df[col].astype(str).map(lambda v: mapping.get(v, unk_val))

    # Ensure order and scale
    X = df.values.astype(float)
    X_scaled = scl.transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=1)  # shape (1,1,features)

    # predict
    prob = float(mdl.predict(X_scaled)[0][0])
    is_fraud = 1 if prob > 0.5 else 0

    result_text = "REJECTED (Fraud Suspected)" if is_fraud == 1 else "APPROVED"
    details = session.get("form", {})

    # clear session form data after use
    session.pop("form", None)

    return render_template("result.html", result=result_text, prob=round(prob, 4), details=details)


# ---------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------
@app.errorhandler(400)
def handle_400(err):
    msg = getattr(err, "description", "Bad Request")
    return render_template("400.html", message=msg), 400


@app.errorhandler(500)
def handle_500(err):
    msg = getattr(err, "description", "Internal Server Error")
    return render_template("500.html", message=msg), 500


if __name__ == "__main__":
    app.run(debug=True)
