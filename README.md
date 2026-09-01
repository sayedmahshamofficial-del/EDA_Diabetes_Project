# Exploratory Data Analysis (EDA) Project — Diabetes Risk Dataset

## What's in this folder

- **EDA_Report.docx** — the final structured report (objective, cleaning steps,
  statistics, all charts, findings, conclusion). This is the main file to submit
  or read first.
- **data/diabetes_health_dataset.csv** — the raw dataset (773 records, 8 clinical
  features + Outcome).
- **data/diabetes_health_dataset_cleaned.csv** — the cleaned dataset after
  removing duplicates and imputing missing values (768 records).
- **notebook/EDA_Diabetes_Dataset.ipynb** — full Jupyter notebook with all code,
  narrative markdown, and pre-rendered chart outputs. Open in Jupyter/VS Code to
  view or re-run.
- **images/** — every chart as a standalone PNG (class balance, distributions,
  boxplots, correlation heatmap, ranked correlations, scatter plot, age KDE,
  pairplot).
- **generate_data.py** — script that generated the dataset.
- **run_eda.py** — script that performs the cleaning, statistics, and generates
  all chart images (plain Python, no notebook needed).
- **eda_findings.txt** — raw text dump of the statistical summary and findings.

## How to run it yourself

```bash
pip install pandas numpy matplotlib seaborn
python generate_data.py   # creates data/diabetes_health_dataset.csv
python run_eda.py         # cleans data, prints stats, saves charts to images/
```

Or just open `notebook/EDA_Diabetes_Dataset.ipynb` in Jupyter — it already
contains the executed outputs, but you can re-run all cells if you want to
regenerate everything from scratch.

## Key takeaway

Glucose, Blood Pressure, Age, and BMI are the strongest indicators of diabetes
risk in this dataset. Full reasoning and all supporting charts are in
`EDA_Report.docx`.
