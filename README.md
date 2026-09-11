# Cardiovascular Risk Prediction — ML Model (Beginner Project)

A beginner-friendly machine learning project that predicts whether a
patient is at risk of **cardiovascular disease** using basic health data
(age, blood pressure, cholesterol, glucose, smoking, alcohol use, activity,
height, weight). Includes a native desktop GUI for live predictions.

---

## 1. Dataset

- **File:** `data/cardio_train.csv`
- **Source:** Kaggle — *"Cardiovascular Disease dataset"* by Svetlana Ulianova
  (https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Size:** 70,000 patient records, 11 features + 1 target column
- **Separator:** `;` (semicolon), not a normal comma

| Column        | Meaning                                              |
|---------------|-------------------------------------------------------|
| age           | Age in **days** (converted to years in the code)     |
| gender        | 1 = female, 2 = male                                  |
| height        | Height in cm                                          |
| weight        | Weight in kg                                          |
| ap_hi         | Systolic blood pressure                                |
| ap_lo         | Diastolic blood pressure                               |
| cholesterol   | 1 = normal, 2 = above normal, 3 = well above normal    |
| gluc          | 1 = normal, 2 = above normal, 3 = well above normal    |
| smoke         | 0 = no, 1 = yes                                        |
| alco          | 0 = no, 1 = yes (alcohol intake)                       |
| active        | 0 = not active, 1 = physically active                  |
| **cardio**    | **Target**: 0 = no disease, 1 = has cardiovascular disease |

---

## 2. Project structure

```
project/
├── data/
│   └── cardio_train.csv          # the dataset
├── cardiovascular_risk_model.py  # trains the models (fully commented)
├── desktop_app.py                # native desktop GUI for live predictions
├── requirements.txt               # Python libraries needed
└── README.md                      # this file
```

After you run `cardiovascular_risk_model.py`, it will also generate:
- `correlation_heatmap.png` — shows which features relate to disease risk
- `confusion_matrix.png` — shows correct vs incorrect predictions
- `roc_curve.png` — shows model's ability to separate the two classes
- `cardio_risk_model.pkl` — the saved trained model
- `cardio_risk_scaler.pkl` — the saved feature scaler (needed to use the model later)

`desktop_app.py` loads those two `.pkl` files, so the training script must be
run at least once before the GUI will work.

---

## 3. How to train the model (VS Code / PyCharm / terminal)

1. **Install Python** (3.9+) if you don't already have it.

2. **Open the project folder** in your editor.

3. **Open a terminal** inside the editor.

4. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

5. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the training script:**
   ```bash
   python cardiovascular_risk_model.py
   ```

7. Check the terminal output for accuracy/AUC scores, and look in the
   project folder for the generated `.png` plots and `.pkl` model files.

---

## 4. Running the desktop GUI

`desktop_app.py` is a native window (built with `ttkbootstrap`/Tkinter, not
a browser page) where you can enter a patient's details and get a live
prediction from the saved model.

```bash
python desktop_app.py
```

Fill in age, gender, height/weight, blood pressure, cholesterol/glucose
level, and the lifestyle toggles, then click **Predict risk** to see the
result, risk probability (as a gauge), and BMI.

> This is a statistical estimate based on historical data, not a medical
> diagnosis.

---

## 5. What the code does, step by step

| Step | What happens | Why it matters |
|------|--------------|-----------------|
| 1 | Import libraries (pandas, sklearn, etc.) | Tools needed for data handling + ML |
| 2 | Load `cardio_train.csv` | Get the raw data into a DataFrame |
| 3 | Clean data: drop `id`, convert age to years, add BMI, remove impossible values (e.g. diastolic > systolic BP) | Real-world data has errors; models learn better from clean data |
| 4 | Quick EDA: class balance + correlation heatmap | Understand the data before modeling |
| 5 | Split into `X` (features) and `y` (target `cardio`) | Model needs to know what it's predicting |
| 6 | Train/test split (80/20) | Test on data the model has never seen, to check real performance |
| 7 | Scale features with `StandardScaler` | Puts all features on a similar numeric scale |
| 8 | Tune a Gradient Boosting model with `GridSearchCV`, then train it alongside Logistic Regression, Decision Tree, and Random Forest | Compare simple vs more powerful models, and search for the best hyperparameters |
| 9 | Pick the best model by ROC-AUC score | AUC is a robust metric for balanced medical classification |
| 10 | Plot confusion matrix + ROC curve | Visualize how well the model performs |
| 11 | Save model + scaler with `joblib` | Reuse the trained model later without retraining (used by `desktop_app.py`) |
| 12 | Example function `predict_cardio_risk(...)` | Shows how to predict risk for one new patient |

---

## 6. Results (from the reference run)

| Model                       | Accuracy | ROC-AUC |
|------------------------------|----------|---------|
| Logistic Regression          | ~0.73    | ~0.79   |
| Decision Tree                 | ~0.73    | ~0.79   |
| Random Forest                 | ~0.73    | ~0.80   |
| **Gradient Boosting (tuned)** | **~0.74**| **~0.80** |

Gradient Boosting (tuned with `GridSearchCV`) was selected as the best
model based on AUC score. Its feature importances show systolic blood
pressure (`ap_hi`) as by far the strongest predictor, followed by age —
both visible in `desktop_app.py`'s predictions and consistent with known
cardiovascular risk factors.

> Note: In medical risk prediction, **recall for the "disease" class**
> matters a lot — missing a real at-risk patient (false negative) is
> more costly than a false alarm. The classification report printed by
> the script shows precision/recall for both classes so you can discuss
> this trade-off in your assignment write-up.

---

## 7. Possible extensions (good for bonus points)

- Try `XGBoost` for potentially higher accuracy.
- Handle class imbalance with `class_weight="balanced"` if needed.
- Package `desktop_app.py` as a standalone `.exe` (e.g. with PyInstaller)
  so it can run without installing Python.
- Add a patient history / batch-prediction mode (load a CSV of multiple
  patients and predict risk for all of them at once).
