"""Cardiovascular risk predictor - native desktop GUI (ttkbootstrap/Tkinter).

Opens as its own window instead of a browser tab. Loads the same
pre-trained model/scaler saved by cardiovascular_risk_model.py
(cardio_risk_model.pkl, cardio_risk_scaler.pkl) and predicts risk from
health details entered in the form.
"""

import tkinter as tk
from tkinter import messagebox

import joblib
import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import DANGER, PRIMARY, SUCCESS

# Column order must match X.columns from training in cardiovascular_risk_model.py
FEATURE_COLUMNS = [
    "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "age_years", "bmi",
]

LEVEL_OPTIONS = ["Normal", "Above normal", "Well above normal"]
LEVEL_TO_CODE = {label: i + 1 for i, label in enumerate(LEVEL_OPTIONS)}

model = joblib.load("cardio_risk_model.pkl")
scaler = joblib.load("cardio_risk_scaler.pkl")


def predict_cardio_risk(*, gender, height, weight, ap_hi, ap_lo,
                         cholesterol, gluc, smoke, alco, active, age_years):
    bmi = weight / ((height / 100) ** 2)
    row = pd.DataFrame([{
        "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
        "age_years": age_years, "bmi": bmi,
    }])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]
    return prediction, probability, bmi


def parse_number(entry, field_name, min_value, max_value, is_int=False):
    raw = entry.get().strip()
    try:
        value = int(raw) if is_int else float(raw)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")
    if not (min_value <= value <= max_value):
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}.")
    return value


class CardioApp(ttk.Window):
    def __init__(self):
        super().__init__(title="Cardiovascular risk predictor", themename="flatly", resizable=(False, False))
        self._build_styles()

        header = ttk.Frame(self, bootstyle=PRIMARY, padding=(28, 20))
        header.pack(fill="x")
        ttk.Label(header, text="Cardiovascular risk predictor", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header, text="Enter your health details for a live risk estimate",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self, padding=28)
        body.pack(fill="both", expand=True)

        fields = ttk.Frame(body)
        fields.pack(fill="x")
        left = ttk.Frame(fields)
        left.pack(side="left", fill="x", expand=True)
        right = ttk.Frame(fields)
        right.pack(side="left", fill="x", expand=True, padx=(24, 0))

        self.age_entry = self._add_entry(left, "Age (years)", "45")
        self.gender_var = self._add_combo(left, "Gender", ["Female", "Male"], "Female")
        self.height_entry = self._add_entry(left, "Height (cm)", "170")
        self.weight_entry = self._add_entry(left, "Weight (kg)", "70")

        self.ap_hi_entry = self._add_entry(right, "Systolic BP (upper)", "120")
        self.ap_lo_entry = self._add_entry(right, "Diastolic BP (lower)", "80")
        self.cholesterol_var = self._add_combo(right, "Cholesterol level", LEVEL_OPTIONS, "Normal")
        self.gluc_var = self._add_combo(right, "Glucose level", LEVEL_OPTIONS, "Normal")

        ttk.Label(body, text="LIFESTYLE", style="SectionLabel.TLabel").pack(anchor="w", pady=(20, 6))
        toggles = ttk.Frame(body)
        toggles.pack(fill="x")
        self.smoke_var = tk.BooleanVar(value=False)
        self.alco_var = tk.BooleanVar(value=False)
        self.active_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toggles, text="Smoker", variable=self.smoke_var, bootstyle="danger-round-toggle"
        ).pack(side="left", padx=(0, 24))
        ttk.Checkbutton(
            toggles, text="Drinks alcohol", variable=self.alco_var, bootstyle="danger-round-toggle"
        ).pack(side="left", padx=(0, 24))
        ttk.Checkbutton(
            toggles, text="Physically active", variable=self.active_var, bootstyle="success-round-toggle"
        ).pack(side="left")

        ttk.Button(
            body, text="Predict risk", command=self.on_predict, bootstyle=PRIMARY, padding=(0, 10),
        ).pack(fill="x", pady=(24, 0))

        self.result_frame = ttk.Frame(body, padding=20, bootstyle="light")
        self.result_meter = ttk.Meter(
            self.result_frame, meter_size=130, amount_total=100, amount_used=0,
            subtext="risk probability", text_right="%", meter_type="semi",
            bootstyle=SUCCESS, stripe_thickness=0, interactive=False,
        )
        self.result_meter.pack(side="left", padx=(0, 20))

        result_text = ttk.Frame(self.result_frame)
        result_text.pack(side="left", fill="both", expand=True, anchor="center")
        self.verdict_var = tk.StringVar(value="")
        self.verdict_label = ttk.Label(result_text, textvariable=self.verdict_var, style="Verdict.TLabel")
        self.verdict_label.pack(anchor="w")
        self.bmi_var = tk.StringVar(value="")
        ttk.Label(result_text, textvariable=self.bmi_var, style="Stat.TLabel").pack(anchor="w", pady=(6, 0))

        ttk.Label(
            body,
            text="Statistical estimate only, not a medical diagnosis. Consult a doctor.",
            style="Disclaimer.TLabel",
        ).pack(anchor="w", pady=(16, 0))

        self.place_window_center()

    def _build_styles(self):
        style = self.style
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground="white", background=style.colors.primary)
        style.configure("Subheader.TLabel", font=("Segoe UI", 10), foreground="white", background=style.colors.primary)
        style.configure("SectionLabel.TLabel", font=("Segoe UI", 9, "bold"), foreground=style.colors.secondary)
        style.configure("FieldLabel.TLabel", font=("Segoe UI", 10))
        style.configure("Verdict.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Stat.TLabel", font=("Segoe UI", 10), foreground=style.colors.secondary)
        style.configure("Disclaimer.TLabel", font=("Segoe UI", 8), foreground=style.colors.secondary)

    def _add_entry(self, parent, label, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, style="FieldLabel.TLabel").pack(anchor="w")
        entry = ttk.Entry(row)
        entry.insert(0, default)
        entry.pack(fill="x", pady=(4, 0))
        return entry

    def _add_combo(self, parent, label, options, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, style="FieldLabel.TLabel").pack(anchor="w")
        var = tk.StringVar(value=default)
        combo = ttk.Combobox(row, textvariable=var, values=options, state="readonly")
        combo.pack(fill="x", pady=(4, 0))
        return var

    def on_predict(self):
        try:
            age_years = parse_number(self.age_entry, "Age", 1, 120, is_int=True)
            height = parse_number(self.height_entry, "Height", 100, 230)
            weight = parse_number(self.weight_entry, "Weight", 20, 250)
            ap_hi = parse_number(self.ap_hi_entry, "Systolic BP", 60, 260, is_int=True)
            ap_lo = parse_number(self.ap_lo_entry, "Diastolic BP", 30, 210, is_int=True)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        prediction, probability, bmi = predict_cardio_risk(
            gender=1 if self.gender_var.get() == "Female" else 2,
            height=height, weight=weight,
            ap_hi=ap_hi, ap_lo=ap_lo,
            cholesterol=LEVEL_TO_CODE[self.cholesterol_var.get()],
            gluc=LEVEL_TO_CODE[self.gluc_var.get()],
            smoke=int(self.smoke_var.get()), alco=int(self.alco_var.get()),
            active=int(self.active_var.get()), age_years=age_years,
        )

        tone = DANGER if prediction == 1 else SUCCESS
        verdict = "At risk of cardiovascular disease" if prediction == 1 else "Low risk / no disease detected"

        self.result_meter.configure(amount_used=round(probability * 100), bootstyle=tone)
        self.verdict_var.set(verdict)
        self.verdict_label.configure(style="Verdict.TLabel", foreground=self.style.colors.get(tone))
        self.bmi_var.set(f"BMI: {bmi:.1f}")
        self.result_frame.pack(fill="x", pady=(20, 0))


if __name__ == "__main__":
    CardioApp().mainloop()
