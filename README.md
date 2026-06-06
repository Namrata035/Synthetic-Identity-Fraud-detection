# Synthetic-Identity-Fraud-detection

# Synthetic Identity Fraud Detection with Loan Approval

## 1. Project Description

Synthetic Identity Fraud Detection with Loan Approval is a deep learning-based banking security system designed to identify fraudulent loan applicants before loan approval. The system uses a hybrid CNN-LSTM model to analyze applicant information and predict whether an identity is legitimate or synthetic. Based on the prediction, the system either approves or rejects the loan application.

---

## 2. Features

* Generates realistic synthetic banking datasets using Faker.
* Detects synthetic identity fraud in loan applications.
* Uses CNN to learn patterns from applicant attributes.
* Uses LSTM to capture behavioral and sequential patterns.
* Provides explainable AI insights using SHAP.
* Flask-based web application for real-time predictions.
* Loan approval recommendation system.
* Visualization of:

  * Accuracy vs Epochs
  * Precision-Recall Curve
  * SHAP Feature Importance
  * Model Comparison

### Workflow

1. Generate synthetic banking data.
2. Perform preprocessing and label encoding.
3. Train CNN-LSTM fraud detection model.
4. Evaluate model performance.
5. Generate SHAP explanations.
6. Predict fraud risk for new applicants.
7. Approve or reject loan application.

---

## 3. Tech Stack

### Programming Language

* Python

### Data Generation

* Faker

### Data Processing

* Pandas
* NumPy

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* Scikit-learn

### Explainable AI

* SHAP

### Data Visualization

* Matplotlib
* Seaborn

### Web Framework

* Flask

### Frontend

* HTML
* CSS

### Development Environment

* Visual Studio Code

---

## 4. How to Run

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Synthetic-fraud-detection-with-loan-approval
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Flask Application

```bash
python app.py
```

### Step 6: Open Browser

```text
http://127.0.0.1:5000
```

Enter applicant details and submit the form to receive fraud prediction and loan approval results.

---

## 5. Project Structure

```text
Synthetic fraud detection with loan approval/
│
├── data/
│   └── synthetic_loan_fraud.csv
│
├── model/
│   ├── cnn_lstm_fraud_model.h5
│   ├── label_encoders.pkl
│   ├── model_columns.pkl
│   └── scaler.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── loan.html
│   ├── result.html
│   ├── 400.html
│   └── 500.html
│
├── app.py
├── data_generator.py
├── preprocessing_label_encoding.py
├── small_cnn_train.py
├── train_model.py
├── graph1_accuracy_vs_epochs.py
├── graph2_precision_recall_curve.py
├── graph3_shap_summary_bar.py
├── graph4_model_comparison.py
├── Graph4_Model_Comparison.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. File Description

| File                             | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| app.py                           | Flask application for fraud prediction and loan approval |
| data_generator.py                | Generates synthetic banking dataset using Faker          |
| preprocessing_label_encoding.py  | Performs preprocessing and label encoding                |
| small_cnn_train.py               | Initial CNN model training                               |
| train_model.py                   | CNN-LSTM model training                                  |
| graph1_accuracy_vs_epochs.py     | Generates training accuracy graph                        |
| graph2_precision_recall_curve.py | Generates precision-recall curve                         |
| graph3_shap_summary_bar.py       | Generates SHAP feature importance graph                  |
| graph4_model_comparison.py       | Compares model performance                               |
| cnn_lstm_fraud_model.h5          | Trained CNN-LSTM model                                   |
| label_encoders.pkl               | Saved label encoders                                     |
| scaler.pkl                       | Saved feature scaler                                     |
| model_columns.pkl                | Stores model input columns                               |
| synthetic_loan_fraud.csv         | Synthetic banking dataset                                |

---

## Output

* Fraudulent Identity → Loan Rejected
* Legitimate Identity → Loan Approved

The system combines deep learning and explainable AI to improve fraud detection accuracy and support secure loan approval decisions.
