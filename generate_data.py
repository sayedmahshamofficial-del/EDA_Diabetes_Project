"""
Generates a realistic synthetic patient health dataset for diabetes-risk
Exploratory Data Analysis. Values follow plausible real-world clinical
ranges and correlations (higher glucose/BMI/age -> higher diabetes risk),
with a bit of missing/noisy data injected on purpose so the EDA has
something real to clean and discuss.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 768  # number of patient records

# --- Base features -----------------------------------------------------
age = rng.integers(21, 81, N)
pregnancies = np.clip(rng.poisson(2.2, N), 0, 15)

# Glucose correlates with age + a risk factor
base_risk = rng.normal(0, 1, N)
glucose = 90 + 0.5 * age + 15 * base_risk + rng.normal(0, 12, N)
glucose = np.clip(glucose, 60, 200).round(0)

blood_pressure = 65 + 0.3 * age + 8 * base_risk + rng.normal(0, 8, N)
blood_pressure = np.clip(blood_pressure, 40, 122).round(0)

bmi = 24 + 4 * base_risk + rng.normal(0, 4, N)
bmi = np.clip(bmi, 15, 55).round(1)

skin_thickness = np.clip(20 + 0.4 * (bmi - 24) + rng.normal(0, 8, N), 0, 60).round(0)
insulin = np.clip(80 + 4 * (glucose - 100) / 10 + rng.normal(0, 60, N), 0, 600).round(0)

diabetes_pedigree = np.clip(rng.gamma(2, 0.2, N), 0.08, 2.4).round(3)

# --- Outcome: logistic function of risk factors -------------------------
z = (
    0.035 * (glucose - 120)
    + 0.045 * (bmi - 28)
    + 0.02 * (age - 33)
    + 0.6 * diabetes_pedigree
    + 0.15 * (pregnancies - 3)
    - 3.5
)
prob = 1 / (1 + np.exp(-z))
outcome = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "Outcome": outcome,
})

# --- Inject realistic messiness for the EDA to uncover ------------------
# A. Some biologically-impossible zeros (common real-world data entry issue)
for col, frac in [("SkinThickness", 0.08), ("Insulin", 0.12), ("BloodPressure", 0.02), ("BMI", 0.01)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = 0

# B. A handful of true missing values (NaN)
for col, frac in [("Glucose", 0.01), ("BMI", 0.015)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# C. A few duplicate rows (data pipeline artifact)
dup_idx = rng.choice(N, size=5, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

df.to_csv("/home/claude/eda_project/data/diabetes_health_dataset.csv", index=False)
print("Saved dataset with shape:", df.shape)
print(df.head())
