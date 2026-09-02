"""
==========================================================================
 CARDIOVASCULAR RISK PREDICTION - BEGINNER FRIENDLY ML MODEL
==========================================================================
Goal   : Predict whether a patient is at risk of cardiovascular disease
         (1 = has disease, 0 = no disease) using basic health data such
         as age, blood pressure, cholesterol, glucose, smoking, etc.

Dataset: cardio_train.csv (70,000 patient records, 11 features + target)
         Source: Kaggle "Cardiovascular Disease dataset" (Svetlana Ulianova)
         File is ';' (semicolon) separated.

How to run in VS Code:
    1. Open this folder in VS Code.
    2. Open a terminal (Terminal -> New Terminal).
    3. Create/activate a virtual environment (optional but recommended):
           python -m venv venv
           venv\\Scripts\\activate      (Windows)
           source venv/bin/activate     (Mac/Linux)
    4. Install requirements:
           pip install -r requirements.txt
    5. Run the script:
           python cardiovascular_risk_model.py
==========================================================================
"""

# --------------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# --------------------------------------------------------------------
import pandas as pd
# pandas -> used to load, clean and manipulate tabular data (like Excel in Python)

import numpy as np
# numpy -> used for numerical operations (arrays, math functions)

import matplotlib.pyplot as plt
# matplotlib -> used to draw graphs/plots (e.g. confusion matrix heatmap)

import seaborn as sns
# seaborn -> built on matplotlib, makes nicer statistical plots (heatmaps, etc.)

from sklearn.model_selection import train_test_split
# train_test_split -> splits our dataset into a "training" part (to teach the
# model) and a "testing" part (to check how well it learned)

from sklearn.preprocessing import StandardScaler
# StandardScaler -> rescales numeric columns so they all have a similar range
# (important because features like 'age' and 'cholesterol' are on very
# different scales, which can confuse some models)

from sklearn.linear_model import LogisticRegression
# LogisticRegression -> a simple, fast, and very interpretable classification
# algorithm - a great baseline model for beginners

from sklearn.tree import DecisionTreeClassifier
# DecisionTreeClassifier -> a model that learns simple "if-else" rules,
# easy to understand and visualize

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# RandomForestClassifier -> builds many decision trees and combines their
# answers (usually more accurate than a single tree)
# GradientBoostingClassifier -> builds trees one at a time, each one
# correcting the mistakes of the previous ones (often more accurate still)

from sklearn.model_selection import GridSearchCV
# GridSearchCV -> tries every combination of a set of hyperparameters and
# keeps the one with the best cross-validated score

from sklearn.metrics import (
    accuracy_score,        # % of correct predictions
    confusion_matrix,      # table showing correct/incorrect predictions per class
    classification_report, # precision, recall, f1-score in one summary
    roc_auc_score,          # measures how well the model separates the 2 classes
    RocCurveDisplay          # plots the ROC curve
)

import joblib
# joblib -> used to save the trained model to a file so we can reuse it later
# without retraining (e.g. inside a web app)


# --------------------------------------------------------------------
# STEP 2: LOAD THE DATASET
# --------------------------------------------------------------------
# The raw CSV uses ';' as the separator, not the usual ',', so we tell
# pandas about it using sep=';'
df = pd.read_csv("data/cardio_train.csv", sep=";")

# Print the shape (rows, columns) so we can confirm the data loaded correctly
print("Dataset shape (rows, columns):", df.shape)

# Show the first 5 rows to visually inspect the data
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Show column data types and check for missing values
print("\nDataset info:")
print(df.info())


# --------------------------------------------------------------------
# STEP 3: CLEAN & PREPROCESS THE DATA
# --------------------------------------------------------------------

# The 'id' column is just a row identifier, it has no predictive value,
# so we drop (remove) it from the dataframe
df = df.drop(columns=["id"])

# The 'age' column is given in DAYS (e.g. 18393 days), which is hard to
# interpret. We convert it into YEARS by dividing by 365 and rounding.
df["age_years"] = (df["age"] / 365).round().astype(int)

# Now that we have age in years, we can drop the original 'age' (in days)
df = df.drop(columns=["age"])

# Create a new, more medically meaningful feature: BMI (Body Mass Index)
# Formula: weight(kg) / height(m)^2   -> height is in cm, so divide by 100
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# --- Remove unrealistic / erroneous entries (common in real-world data) ---

# Blood pressure: diastolic (ap_lo) should never be higher than
# systolic (ap_hi). Rows where this happens are data entry errors.
df = df[df["ap_hi"] >= df["ap_lo"]]

# Keep only physically realistic blood pressure ranges
# (systolic 80-250, diastolic 40-200 mmHg covers virtually all real cases)
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]

# Keep only realistic height (cm) and weight (kg) values
df = df[(df["height"] >= 130) & (df["height"] <= 210)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]

# Reset the row index after all the filtering above (keeps things tidy)
df = df.reset_index(drop=True)

print("\nShape after cleaning:", df.shape)


# --------------------------------------------------------------------
# STEP 4: QUICK EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------

# Check how balanced the target classes are (important - if one class
# dominates, accuracy alone can be misleading)
print("\nTarget class balance (0 = no disease, 1 = disease):")
print(df["cardio"].value_counts(normalize=True))

# Plot a correlation heatmap to see which features relate most strongly
# to the target variable 'cardio'
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")   # saves the plot as an image file
plt.close()                              # closes the figure to free memory


# --------------------------------------------------------------------
# STEP 5: SPLIT FEATURES (X) AND TARGET (y)
# --------------------------------------------------------------------

# X = all the input columns the model will learn from (everything except
# the target 'cardio')
X = df.drop(columns=["cardio"])

# y = the column we want to predict (0 = healthy, 1 = at risk)
y = df["cardio"]


# --------------------------------------------------------------------
# STEP 6: TRAIN-TEST SPLIT
# --------------------------------------------------------------------

# Split data: 80% for training the model, 20% for testing it on unseen data
# random_state=42 makes the split reproducible (same split every run)
# stratify=y keeps the same 0/1 disease ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# --------------------------------------------------------------------
# STEP 7: FEATURE SCALING
# --------------------------------------------------------------------

# Create a scaler object that will standardize numeric features
# (mean = 0, standard deviation = 1). Needed mainly for Logistic Regression.
scaler = StandardScaler()

# Fit the scaler ONLY on training data (learn mean/std), then transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the SAME scaler (do not re-fit, to avoid
# "data leakage" from the test set into the model)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------------------------
# STEP 8: TRAIN MULTIPLE MODELS
# --------------------------------------------------------------------

# Before comparing models, tune a Gradient Boosting model with
# GridSearchCV: it tries every combination of the parameters below and
# keeps the one with the best cross-validated ROC-AUC score.
gb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}

gb_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=gb_param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
)
gb_search.fit(X_train_scaled, y_train)

print("\nBest Gradient Boosting parameters:", gb_search.best_params_)

# We will train several beginner-friendly models and compare their
# performance. A dictionary lets us loop through them cleanly.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    ),
    "Gradient Boosting (tuned)": gb_search.best_estimator_,
}

# Dictionary to store the results of every model for comparison later
results = {}

# Loop through each model: train it, test it, and store its accuracy
for name, model in models.items():

    # Logistic Regression benefits from scaled data; tree-based models don't
    # strictly need it, but using scaled data for all keeps the code simple
    # and doesn't hurt tree-based model performance.
    model.fit(X_train_scaled, y_train)          # train the model
    y_pred = model.predict(X_test_scaled)         # predict on unseen test data
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

    acc = accuracy_score(y_test, y_pred)          # calculate accuracy
    auc = roc_auc_score(y_test, y_proba)          # calculate ROC-AUC score

    results[name] = {"model": model, "accuracy": acc, "auc": auc}

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# --------------------------------------------------------------------
# STEP 9: PICK THE BEST MODEL (based on ROC-AUC)
# --------------------------------------------------------------------

# Find the model name with the highest AUC score using max() + a lambda key
best_model_name = max(results, key=lambda name: results[name]["auc"])
best_model = results[best_model_name]["model"]

print(f"\nBest performing model: {best_model_name}")
print(f"AUC Score: {results[best_model_name]['auc']:.4f}")


# --------------------------------------------------------------------
# STEP 10: VISUALIZE RESULTS FOR THE BEST MODEL
# --------------------------------------------------------------------

# Get predictions from the best model again for plotting
y_pred_best = best_model.predict(X_test_scaled)

# Compute the confusion matrix: rows = actual class, columns = predicted class
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Plot the ROC curve to visualize the model's ability to distinguish classes
RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
plt.title(f"ROC Curve - {best_model_name}")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

print("\nSaved plots: correlation_heatmap.png, confusion_matrix.png, roc_curve.png")


# --------------------------------------------------------------------
# STEP 11: SAVE THE TRAINED MODEL AND SCALER FOR FUTURE USE
# --------------------------------------------------------------------

# Save the best model to a .pkl file so it can be loaded later without
# retraining (e.g. in a web app or another script)
joblib.dump(best_model, "cardio_risk_model.pkl")

# Save the scaler too - new/unseen data must be scaled the same way
joblib.dump(scaler, "cardio_risk_scaler.pkl")

print("\nModel and scaler saved as 'cardio_risk_model.pkl' and 'cardio_risk_scaler.pkl'")


# --------------------------------------------------------------------
# STEP 12: EXAMPLE - PREDICT RISK FOR A NEW PATIENT
# --------------------------------------------------------------------

def predict_cardio_risk(gender, height, weight, ap_hi, ap_lo,
                         cholesterol, gluc, smoke, alco, active, age_years):
    """
    Takes a new patient's details and returns:
      - prediction: 0 (no disease) or 1 (at risk)
      - probability: how confident the model is (0 to 1)

    Parameter meaning (matches the original dataset encoding):
      gender      : 1 = female, 2 = male
      height      : in cm
      weight      : in kg
      ap_hi       : systolic blood pressure
      ap_lo       : diastolic blood pressure
      cholesterol : 1 = normal, 2 = above normal, 3 = well above normal
      gluc        : 1 = normal, 2 = above normal, 3 = well above normal
      smoke       : 0 = non-smoker, 1 = smoker
      alco        : 0 = no alcohol, 1 = consumes alcohol
      active      : 0 = not physically active, 1 = physically active
      age_years   : patient's age in years
    """
    # Compute BMI the same way we did during training
    bmi = weight / ((height / 100) ** 2)

    # Build a single-row dataframe with columns in the SAME order as X_train
    new_data = pd.DataFrame([{
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age_years,
        "bmi": bmi,
    }])[X.columns]  # reorder columns to exactly match training data

    # Scale the new patient's data using the SAME scaler used in training
    new_data_scaled = scaler.transform(new_data)

    # Predict class (0 or 1) and probability of being at risk
    prediction = best_model.predict(new_data_scaled)[0]
    probability = best_model.predict_proba(new_data_scaled)[0][1]

    return prediction, probability


# Example usage: a 55-year-old male with high blood pressure and cholesterol
pred, prob = predict_cardio_risk(
    gender=2, height=170, weight=85, ap_hi=150, ap_lo=95,
    cholesterol=3, gluc=1, smoke=1, alco=0, active=0, age_years=55
)

print("\n--- Example Prediction ---")
print("Predicted class   :", "At Risk" if pred == 1 else "No Disease")
print(f"Risk probability  : {prob:.2%}")


# --------------------------------------------------------------------
# STEP 13: INTERACTIVE USER INPUT -> LIVE PREDICTION
# --------------------------------------------------------------------
# This section lets a real person type in their own health details in
# the terminal/console, and the trained model predicts their risk.

def get_int_input(prompt, valid_range=None, valid_choices=None):
    """
    Helper function that keeps asking the user for input until they
    type a valid whole number. Prevents the program from crashing if
    someone types letters or an out-of-range value by mistake.

    prompt        : text shown to the user
    valid_range   : optional (min, max) tuple to restrict numeric range
    valid_choices : optional set/list of allowed exact values (e.g. {1,2})
    """
    while True:  # keep looping until we get valid input
        raw = input(prompt).strip()  # read text typed by user, remove spaces
        try:
            value = int(raw)  # try converting text to a whole number
        except ValueError:
            print("  -> Please enter a valid whole number.")
            continue  # go back to the start of the loop and ask again

        if valid_choices is not None and value not in valid_choices:
            print(f"  -> Please enter one of: {sorted(valid_choices)}")
            continue

        if valid_range is not None and not (valid_range[0] <= value <= valid_range[1]):
            print(f"  -> Please enter a number between {valid_range[0]} and {valid_range[1]}.")
            continue

        return value  # input passed all checks, return it


def get_float_input(prompt, valid_range=None):
    """Same idea as get_int_input, but allows decimal numbers (e.g. weight 72.5)."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("  -> Please enter a valid number.")
            continue

        if valid_range is not None and not (valid_range[0] <= value <= valid_range[1]):
            print(f"  -> Please enter a number between {valid_range[0]} and {valid_range[1]}.")
            continue

        return value


def run_interactive_prediction():
    """
    Asks the user a series of questions in the terminal, collects their
    health details, then uses the trained model to predict cardiovascular
    disease risk and prints the result.
    """
    print("\n" + "=" * 60)
    print(" CARDIOVASCULAR RISK PREDICTOR - Enter your details below")
    print("=" * 60)

    # Collect each feature one by one, with validation
    age_years = get_int_input("Age (years, e.g. 45): ", valid_range=(1, 120))

    gender = get_int_input("Gender (1 = Female, 2 = Male): ", valid_choices={1, 2})

    height = get_float_input("Height in cm (e.g. 170): ", valid_range=(100, 230))

    weight = get_float_input("Weight in kg (e.g. 70): ", valid_range=(20, 250))

    ap_hi = get_int_input("Systolic Blood Pressure / upper number (e.g. 120): ",
                           valid_range=(60, 260))

    ap_lo = get_int_input("Diastolic Blood Pressure / lower number (e.g. 80): ",
                           valid_range=(30, 210))

    cholesterol = get_int_input(
        "Cholesterol level (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): ",
        valid_choices={1, 2, 3}
    )

    gluc = get_int_input(
        "Glucose level (1 = Normal, 2 = Above Normal, 3 = Well Above Normal): ",
        valid_choices={1, 2, 3}
    )

    smoke = get_int_input("Do you smoke? (0 = No, 1 = Yes): ", valid_choices={0, 1})

    alco = get_int_input("Do you drink alcohol? (0 = No, 1 = Yes): ", valid_choices={0, 1})

    active = get_int_input("Are you physically active? (0 = No, 1 = Yes): ", valid_choices={0, 1})

    # Feed all the collected answers into our prediction function
    prediction, probability = predict_cardio_risk(
        gender=gender, height=height, weight=weight,
        ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol,
        gluc=gluc, smoke=smoke, alco=alco, active=active,
        age_years=age_years
    )

    # Display the result in a clear, readable way
    print("\n" + "-" * 60)
    print(" PREDICTION RESULT")
    print("-" * 60)
    if prediction == 1:
        print(" Result       : ⚠️  AT RISK of Cardiovascular Disease")
    else:
        print(" Result       : ✅  LOW RISK / No Disease Detected")
    print(f" Confidence   : {probability:.2%} probability of disease")
    print("-" * 60)
    print(" Note: This is a machine learning estimate based on statistical")
    print(" patterns in historical data, NOT a medical diagnosis. Please")
    print(" consult a doctor for an actual medical evaluation.")
    print("-" * 60)


# --------------------------------------------------------------------
# STEP 14: MAIN LOOP - let the user run predictions repeatedly
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Keep asking the user if they want to check another prediction,
    # so they don't have to re-run the whole script (and retrain the
    # model) every single time.
    while True:
        run_interactive_prediction()

        again = input("\nCheck another patient? (y/n): ").strip().lower()
        if again != "y":
            print("\nExiting. Thank you!")
            break  # exit the while loop, ending the program
