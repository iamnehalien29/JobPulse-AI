
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Load new dataset ----------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "job_postings_2026_sample.csv"
)

df = pd.read_csv(DATA_PATH)

# ---------- Basic inspection ----------

print("\n========== JOBPULSE AI: NEW DATASET ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nData information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique companies:")
print(df["company"].nunique())

print("\nCountry distribution:")
print(df["country"].value_counts())

print("\nTop job categories:")
print(df["category"].value_counts().head(10))

# ---------- India-only investigation ----------

print("\n========== INDIA JOB MARKET ==========")

india_df = df[df["country"] == "IN"].copy()

print("\nIndia dataset shape:")
print(india_df.shape)

print("\nTop 15 hiring cities:")
print(india_df["city"].value_counts().head(15))

print("\nJob categories in India:")
print(india_df["category"].value_counts().head(15))

print("\nSeniority distribution:")
print(india_df["seniority"].value_counts())

print("\nRemote work distribution:")
print(india_df["remote"].value_counts(dropna=False))

# Convert dates for analysis
india_df["posted_date"] = pd.to_datetime(
    india_df["posted_date"],
    errors="coerce"
)

print("\nIndia date range:")
print(india_df["posted_date"].min(), "to", india_df["posted_date"].max())

print("\nInvalid dates:")
print(india_df["posted_date"].isna().sum())

# Salary coverage
print("\nIndia salary records:")
print(india_df["salary_annual_usd"].notna().sum())

print("\nIndia salary missing percentage:")
print(india_df["salary_annual_usd"].isna().mean() * 100)

print("\nIndia salary statistics (USD):")
print(india_df["salary_annual_usd"].describe())

print("\n========== INDIA SALARY OUTLIERS ==========")

salary = india_df["salary_annual_usd"].dropna()

print("\nLowest 10 salaries:")
print(salary.nsmallest(10))

print("\nHighest 10 salaries:")
print(salary.nlargest(10))

print("\nSalary percentiles:")
print(salary.quantile([0.10, 0.25, 0.50, 0.75, 0.90]))

print("\nSalary records by month:")
print(
    india_df.groupby(
        india_df["posted_date"].dt.to_period("M")
    )["salary_annual_usd"].count()
)

# ---------- Salary record investigation ----------

print("\n========== SALARY RECORD INVESTIGATION ==========")

salary_records = india_df[
    india_df["salary_annual_usd"].notna()
].copy()

columns_to_inspect = [
    "posted_date",
    "title",
    "company",
    "city",
    "seniority",
    "experience_years",
    "salary_annual_usd"
]

print("\n15 lowest salary records:")
print(
    salary_records
    .sort_values("salary_annual_usd")
    [columns_to_inspect]
    .head(15)
    .to_string(index=False)
)

print("\n15 highest salary records:")
print(
    salary_records
    .sort_values("salary_annual_usd", ascending=False)
    [columns_to_inspect]
    .head(15)
    .to_string(index=False)
)

# ---------- Monthly posting distribution ----------

print("\n========== MONTHLY POSTING DISTRIBUTION ==========")

print(
    india_df.groupby(
        india_df["posted_date"].dt.to_period("M")
    ).size()
)

# ---------- Salary coverage by month ----------

print("\n========== MONTHLY SALARY COVERAGE ==========")

monthly_salary = india_df.groupby(
    india_df["posted_date"].dt.to_period("M")
)["salary_annual_usd"].agg(
    total_records="size",
    salary_available="count"
)

monthly_salary["salary_coverage_percent"] = (
    monthly_salary["salary_available"]
    / monthly_salary["total_records"]
    * 100
).round(2)

print(monthly_salary)
# ---------- Statistical salary outlier detection ----------

print("\n========== STATISTICAL SALARY OUTLIERS ==========")

Q1 = salary_records["salary_annual_usd"].quantile(0.25)
Q3 = salary_records["salary_annual_usd"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

salary_records["is_outlier"] = (
    (salary_records["salary_annual_usd"] < lower_bound)
    |
    (salary_records["salary_annual_usd"] > upper_bound)
)

print("\nOutlier count:")
print(salary_records["is_outlier"].sum())

print("\nOutlier percentage:")
print(
    round(
        salary_records["is_outlier"].mean() * 100,
        2
    ),
    "%"
)

print("\nHighest salary outliers:")
print(
    salary_records[salary_records["is_outlier"]]
    .sort_values("salary_annual_usd", ascending=False)
    [["title", "company", "city", "salary_annual_usd"]]
    .head(20)
    .to_string(index=False)
)

print("Total salary records:", len(salary_records))
# ---------- Salary quality classification ----------

print("\n========== SALARY QUALITY CLASSIFICATION ==========")

conditions = [
    salary_records["salary_annual_usd"] < 3000,
    salary_records["salary_annual_usd"] > 54600
]

choices = [
    "Suspiciously Low",
    "High - Verify"
]

salary_records["salary_flag"] = np.select(
    conditions,
    choices,
    default="Within IQR Bounds"
)

print("\nSalary classification:")
print(salary_records["salary_flag"].value_counts())

print("\nExamples of suspiciously low salaries:")
print(
    salary_records[
        salary_records["salary_flag"] == "Suspiciously Low"
    ][["title", "company", "salary_annual_usd"]]
    .head(10)
    .to_string(index=False)
)