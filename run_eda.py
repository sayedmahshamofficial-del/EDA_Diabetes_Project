import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

IMG = "/home/claude/eda_project/images"
DATA = "/home/claude/eda_project/data/diabetes_health_dataset.csv"

df = pd.read_csv(DATA)

# ------------------------------------------------------------------
# 1. Basic cleaning
# ------------------------------------------------------------------
before_rows = len(df)
df = df.drop_duplicates()
after_dedup = len(df)

# Columns where 0 is biologically implausible -> treat as missing
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
missing_before = df[zero_as_missing].isin([0]).sum() + df[zero_as_missing].isna().sum()

for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)

missing_summary = df.isna().sum()

# Impute with median (robust to outliers) grouped by Outcome
for col in zero_as_missing + ["Glucose", "BMI"]:
    df[col] = df.groupby("Outcome")[col].transform(lambda s: s.fillna(s.median()))

df["Outcome_Label"] = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})

summary_stats = df.describe().T
corr = df.drop(columns=["Outcome_Label"]).corr()
outcome_corr = corr["Outcome"].drop("Outcome").sort_values(key=abs, ascending=False)

# ------------------------------------------------------------------
# 2. Save cleaning + stats summary as text/CSV for the report
# ------------------------------------------------------------------
with open("/home/claude/eda_project/eda_findings.txt", "w") as f:
    f.write("EDA FINDINGS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Raw rows: {before_rows} | After removing duplicates: {after_dedup} "
            f"({before_rows - after_dedup} duplicate rows removed)\n\n")
    f.write("Missing / implausible-zero values found per column (before cleaning):\n")
    f.write(missing_before.to_string() + "\n\n")
    f.write("Missing values remaining after converting zeros to NaN (before imputation):\n")
    f.write(missing_summary.to_string() + "\n\n")
    f.write("Cleaning approach: implausible zeros in Glucose, BloodPressure, SkinThickness,\n")
    f.write("Insulin, and BMI were treated as missing and imputed using the median value\n")
    f.write("within each Outcome group (diabetic / non-diabetic), which preserves the\n")
    f.write("distributional differences between the two groups better than a global median.\n\n")
    f.write("Final dataset shape: " + str(df.shape) + "\n\n")
    f.write("Descriptive statistics:\n")
    f.write(summary_stats.to_string() + "\n\n")
    f.write("Correlation of each feature with Outcome (diabetes), sorted by strength:\n")
    f.write(outcome_corr.to_string() + "\n\n")
    f.write("Class balance:\n")
    f.write(df["Outcome_Label"].value_counts().to_string() + "\n")

df.to_csv("/home/claude/eda_project/data/diabetes_health_dataset_cleaned.csv", index=False)

print("Cleaning + stats complete.")
print(outcome_corr)

# ------------------------------------------------------------------
# 3. Visualizations
# ------------------------------------------------------------------
palette = {"No Diabetes": "#4C72B0", "Diabetes": "#DD8452"}

# 3a. Class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Outcome_Label"].value_counts()
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False, ax=ax)
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
ax.set_title("Class Balance: Diabetes Outcome")
ax.set_ylabel("Number of Patients")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{IMG}/01_class_balance.png")
plt.close()

# 3b. Histograms / distributions of numeric features
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(col)
fig.suptitle("Distribution of Each Health Feature", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/02_feature_distributions.png")
plt.close()

# 3c. Boxplots by outcome (spot outliers + separation)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.boxplot(data=df, x="Outcome_Label", y=col, hue="Outcome_Label",
                palette=palette, legend=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Feature Spread by Diabetes Outcome (outlier check)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/03_boxplots_by_outcome.png")
plt.close()

# 3d. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix of Health Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/04_correlation_heatmap.png")
plt.close()

# 3e. Top correlated features with Outcome (bar chart)
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#DD8452" if v > 0 else "#4C72B0" for v in outcome_corr.values]
sns.barplot(x=outcome_corr.values, y=outcome_corr.index, hue=outcome_corr.index,
            palette=colors, legend=False, ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlation of Each Feature with Diabetes Outcome")
ax.set_xlabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(f"{IMG}/05_outcome_correlation_ranked.png")
plt.close()

# 3f. Glucose vs BMI scatter, colored by outcome (key influencing factors)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Outcome_Label",
                 palette=palette, alpha=0.7, ax=ax)
ax.set_title("Glucose vs BMI, Colored by Diabetes Outcome")
plt.tight_layout()
plt.savefig(f"{IMG}/06_glucose_vs_bmi_scatter.png")
plt.close()

# 3g. Age distribution by outcome (KDE)
fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=df, x="Age", hue="Outcome_Label", fill=True, common_norm=False,
            palette=palette, alpha=0.5, ax=ax)
ax.set_title("Age Distribution: Diabetic vs Non-Diabetic Patients")
plt.tight_layout()
plt.savefig(f"{IMG}/07_age_distribution_by_outcome.png")
plt.close()

# 3h. Pairplot of top influencing factors
top_feats = list(outcome_corr.index[:4]) + ["Outcome_Label"]
pp = sns.pairplot(df[top_feats], hue="Outcome_Label", palette=palette,
                   diag_kind="kde", plot_kws={"alpha": 0.6, "s": 25})
pp.fig.suptitle("Pairwise Relationships of Top Risk Factors", y=1.02, fontsize=14, fontweight="bold")
pp.savefig(f"{IMG}/08_pairplot_top_factors.png")
plt.close()

print("All visualizations saved to", IMG)
