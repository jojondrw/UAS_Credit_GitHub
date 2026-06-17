import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class CreditScoringInference:
    def __init__(self):
        # Load preprocessing + model yang sudah dilatih
        self.preprocessing = joblib.load(ARTIFACT_DIR / "preprocessing.pkl")
        self.model = joblib.load(ARTIFACT_DIR / "best_model.pkl")

    def predict(self, input_dict):
        # Bungkus input 1 nasabah jadi DataFrame 1 baris
        df = pd.DataFrame([input_dict])

        # Transform pakai preprocessing yang sama dengan training
        X = self.preprocessing.transform(df)

        # Prediksi label + probabilitas
        label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # Pasangkan tiap kelas dengan probabilitasnya
        proba_dict = {cls: float(p) for cls, p in zip(self.model.classes_, proba)}

        return {"prediction": label, "probabilities": proba_dict}


if __name__ == "__main__":
    # Smoke test 1 nasabah
    sample = {
        'Age': 35, 'Annual_Income': 20364.57, 'Monthly_Inhand_Salary': 1626.05,
        'Num_Bank_Accounts': 6, 'Num_Credit_Card': 8, 'Interest_Rate': 32,
        'Num_of_Loan': 3, 'Delay_from_due_date': 23, 'Num_of_Delayed_Payment': 9,
        'Changed_Credit_Limit': 10.3, 'Num_Credit_Inquiries': 11, 'Outstanding_Debt': 2500.04,
        'Credit_Utilization_Ratio': 27.58, 'Credit_History_Age': 116, 'Total_EMI_per_month': 26.17,
        'Amount_invested_monthly': 92.52, 'Monthly_Balance': 303.92,
        'Credit_Mix': 'Standard', 'Month': 'August', 'Occupation': 'Journalist',
        'Payment_of_Min_Amount': 'Yes', 'Payment_Behaviour': 'High_spent_Small_value_payments',
        'Auto_Loan': 0, 'Credit_Builder_Loan': 0, 'Debt_Consolidation_Loan': 0,
        'Home_Equity_Loan': 0, 'Mortgage_Loan': 1, 'Payday_Loan': 0,
        'Personal_Loan': 1, 'Student_Loan': 1
    }

    infer = CreditScoringInference()
    result = infer.predict(sample)
    print("Prediksi    :", result["prediction"])
    print("Probabilitas:", result["probabilities"])