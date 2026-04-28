import os
import numpy as np
import pandas as pd


def _clamp(values, min_value=0.0, max_value=1.0):
    return np.clip(values, min_value, max_value)


def generate_hr_hiring_data(n_rows, rng):
    years_experience = rng.integers(0, 31, size=n_rows)
    education_levels = np.array(["HighSchool", "Bachelors", "Masters", "PhD"])
    education = rng.choice(education_levels, size=n_rows, p=[0.25, 0.45, 0.25, 0.05])
    age = rng.integers(21, 66, size=n_rows)
    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.55, 0.45])
    interview_score = rng.integers(0, 101, size=n_rows)

    education_score = np.select(
        [education == "HighSchool", education == "Bachelors", education == "Masters", education == "PhD"],
        [0.05, 0.12, 0.18, 0.22],
        default=0.1,
    )

    base_score = (
        (years_experience / 30.0) * 0.4
        + (interview_score / 100.0) * 0.5
        + education_score
    )

    bias_penalty = np.zeros(n_rows)
    bias_penalty += np.where(age > 50, 0.2, 0.0)
    bias_penalty += np.where(gender == "Female", 0.15, 0.0)

    hire_probability = _clamp(base_score - bias_penalty)
    hired = rng.random(n_rows) < hire_probability

    return pd.DataFrame(
        {
            "Years_Experience": years_experience,
            "Education": education,
            "Age": age,
            "Gender": gender,
            "Interview_Score": interview_score,
            "Hired": hired.astype(int),
        }
    )


def generate_finance_loan_data(n_rows, rng):
    income = rng.lognormal(mean=10.5, sigma=0.5, size=n_rows)
    income = np.clip(income, 20000, 200000).round(2)
    credit_score = rng.integers(300, 851, size=n_rows)
    loan_amount = rng.integers(2000, 50001, size=n_rows)
    zip_codes = np.array(["Z1", "Z2", "Z3", "Z4", "Z5"])
    zip_code = rng.choice(zip_codes, size=n_rows, p=[0.25, 0.2, 0.25, 0.15, 0.15])
    marital_status = rng.choice(
        ["Married", "Single", "Divorced"], size=n_rows, p=[0.5, 0.35, 0.15]
    )

    loan_ratio = loan_amount / income
    base_score = (
        (income / 200000.0) * 0.35
        + ((credit_score - 300) / 550.0) * 0.55
        - loan_ratio * 0.2
    )

    bias_penalty = np.zeros(n_rows)
    bias_penalty += np.where(np.isin(zip_code, ["Z2", "Z4"]), 0.2, 0.0)
    bias_penalty += np.where(marital_status == "Single", 0.15, 0.0)

    approval_probability = _clamp(base_score - bias_penalty)
    loan_approved = rng.random(n_rows) < approval_probability

    return pd.DataFrame(
        {
            "Income": income,
            "Credit_Score": credit_score,
            "Loan_Amount": loan_amount,
            "Zip_Code": zip_code,
            "Marital_Status": marital_status,
            "Loan_Approved": loan_approved.astype(int),
        }
    )


def generate_medical_triage_data(n_rows, rng):
    symptoms_severity = rng.integers(1, 11, size=n_rows)
    blood_pressure = rng.integers(80, 181, size=n_rows)
    bmi = rng.uniform(15, 40, size=n_rows).round(1)
    income_bracket = rng.choice(["Low", "Medium", "High"], size=n_rows, p=[0.4, 0.4, 0.2])
    race_proxy = rng.choice(["GroupA", "GroupB", "GroupC"], size=n_rows, p=[0.45, 0.35, 0.2])

    bp_risk = np.where(blood_pressure >= 140, 0.2, 0.0)
    bmi_risk = np.where(bmi >= 30, 0.1, 0.0)

    base_score = (
        (symptoms_severity / 10.0) * 0.6 + bp_risk + bmi_risk
    )

    bias_penalty = np.where(income_bracket == "Low", 0.2, 0.0)
    care_probability = _clamp(base_score - bias_penalty)
    immediate_care = rng.random(n_rows) < care_probability

    return pd.DataFrame(
        {
            "Symptoms_Severity": symptoms_severity,
            "Blood_Pressure": blood_pressure,
            "BMI": bmi,
            "Income_Bracket": income_bracket,
            "Race_Proxy": race_proxy,
            "Immediate_Care_Approved": immediate_care.astype(int),
        }
    )


def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    n_rows = 10000

    hr_df = generate_hr_hiring_data(n_rows, rng)
    finance_df = generate_finance_loan_data(n_rows, rng)
    medical_df = generate_medical_triage_data(n_rows, rng)

    hr_df.to_csv(os.path.join(output_dir, "hr_hiring_data.csv"), index=False)
    finance_df.to_csv(os.path.join(output_dir, "finance_loan_data.csv"), index=False)
    medical_df.to_csv(os.path.join(output_dir, "medical_triage_data.csv"), index=False)

    print("Generated datasets in:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
