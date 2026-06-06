# data_generator.py
import pandas as pd
import random
from faker import Faker
import os
import numpy as np
import string

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

def generate_synthetic_data(n=10000):
    os.makedirs("data", exist_ok=True)
    data = []
    for _ in range(n):
        # fraud probability 8%
        is_fraud = int(random.random() < 0.08)

        # Identity / KYC
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=65).isoformat()
        gender = random.choice(["Male","Female","Other"])
        email = fake.email()
        # make phone 10 digits
        phone = fake.msisdn()[:10]
        address = fake.address().replace("\n", ", ")
        
        # CUSTOM pattern: 5 letters + 5 digits (e.g., ABCDE12345)
        pan = ''.join(random.choices(string.ascii_uppercase, k=5)) + \
              ''.join(random.choices(string.digits, k=5))

        aadhaar = str(fake.random_number(digits=12, fix_len=True))

        # Loan Application & financial
        loan_type = random.choice(["Personal","Home","Auto","Education","Business"])
        loan_amount = random.randint(50000, 2500000)
        income = random.randint(150000, 2000000)
        # For frauds we sometimes make credit score low or missing
        credit_score = (random.randint(300, 850) if not is_fraud else random.choice([None, random.randint(300, 550)]))
        debt_to_income_ratio = round(loan_amount / (income + 1), 2)
        loan_purpose = random.choice(["Renovation","Marriage","Travel","Medical","Investment","Business"])
        application_channel = random.choice(["Online","Branch","Agent"])
        repayment_status = random.choice(["On-time","Delayed","Defaulted"])
        loan_approval_status = random.choice(["Approved","Rejected"])

        # device / network / behavior
        ip_address = fake.ipv4_public()
        device_id = fake.uuid4()
        user_agent = random.choice(["Chrome","Firefox","Edge","Safari","Opera"])
        city = fake.city()
        state = fake.state()
        geo_lat = round(random.uniform(8.0, 37.0), 6)
        geo_long = round(random.uniform(68.0, 97.0), 6)
        form_fill_time_seconds = random.randint(8, 600) if not is_fraud else random.randint(5, 50)
        login_time_hour = random.randint(0,23)
        failed_otp_count = random.randint(0,5)
        num_txn_last_30d = random.randint(0,100)

        # graph-like derived features
        shared_phone_count = random.randint(1,3) if is_fraud else 1
        shared_device_count = random.randint(1,4) if is_fraud else 1
        shared_ip_count = random.randint(1,6) if is_fraud else 1
        email_domain_count = random.randint(1,10)
        aadhaar_verified = (False if is_fraud and random.random() < 0.6 else True)
        pan_verified = (False if is_fraud and random.random() < 0.6 else True)
        phone_verified = (False if is_fraud and random.random() < 0.3 else True)
        doc_match_score = round(random.uniform(0.1,0.7),2) if is_fraud else round(random.uniform(0.6,1.0),2)

        data.append({
            "name": name,
            "dob": dob,
            "gender": gender,
            "email": email,
            "phone": phone,
            "address": address,
            "pan": pan,
            "aadhaar": aadhaar,
            "loan_type": loan_type,
            "loan_amount": loan_amount,
            "income": income,
            "credit_score": credit_score,
            "debt_to_income_ratio": debt_to_income_ratio,
            "loan_purpose": loan_purpose,
            "application_channel": application_channel,
            "repayment_status": repayment_status,
            "loan_approval_status": loan_approval_status,
            "ip_address": ip_address,
            "device_id": device_id,
            "user_agent": user_agent,
            "city": city,
            "state": state,
            "geo_lat": geo_lat,
            "geo_long": geo_long,
            "form_fill_time_seconds": form_fill_time_seconds,
            "login_time_hour": login_time_hour,
            "failed_otp_count": failed_otp_count,
            "num_txn_last_30d": num_txn_last_30d,
            "shared_phone_count": shared_phone_count,
            "shared_device_count": shared_device_count,
            "shared_ip_count": shared_ip_count,
            "email_domain_count": email_domain_count,
            "aadhaar_verified": aadhaar_verified,
            "pan_verified": pan_verified,
            "phone_verified": phone_verified,
            "doc_match_score": doc_match_score,
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_loan_fraud.csv", index=False)
    print(f"[✓] Saved {n:,} records to data/synthetic_loan_fraud.csv")

if __name__ == "__main__":
    generate_synthetic_data(10000)
