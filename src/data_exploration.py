import pandas as pd
from pathlib import Path

# ---------- Load data ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "indian_tech_career_intelligence_2026.csv"

df = pd.read_csv(DATA_PATH)
df["posted_date"] = pd.to_datetime(df["posted_date"], format="%d-%m-%Y", errors="coerce")

print(df.shape)
print(df.columns.tolist())

# ---------- Reliability check ----------
print("\n========== DATASET RELIABILITY CHECK ==========")

print("\nCountries:", df["location_country"].unique())
print("\nSalary currencies:", df["salary_currency"].unique())
print("\nSource platforms:", df["source_platform"].unique())

print("\nJob categories:")
print(df["job_category"].value_counts().head(10))

print("\nInvalid salary records:", (df["salary_minimum"] > df["salary_maximum"]).sum())
print("Unique companies:", df["company_name"].nunique())

# ---------- Data quality ----------
print("\n========== DATA QUALITY ==========")

nulls = df.isnull().sum()
print("\nMissing values:")
print(nulls[nulls > 0])

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSalary summary:")
print(df[["salary_minimum", "salary_maximum"]].describe())

print("\nTop cities:")
print(df["location_city"].value_counts().head(10))

print("\nExperience levels:")
print(df["experience_level"].value_counts())

print("\nDate range:", df["posted_date"].min(), "to", df["posted_date"].max())

print("\nSample skills:")
print(df["skills_extracted"].head(3).tolist())

# ---------- Is there signal in salary? ----------
print("\n========== SALARY SIGNAL CHECK ==========")

print("\nAvg max salary by category:")
print(df.groupby("job_category")["salary_maximum"].mean().round())

print("\nAvg max salary by experience level:")
print(df.groupby("experience_level")["salary_maximum"].mean().round())

print("\nAvg max salary by city:")
print(df.groupby("location_city")["salary_maximum"].mean().round())

print("\nUnique skill strings:", df["skills_extracted"].nunique())

print("\nProbation_period null rate by employment type:")
print(
    df.groupby("employment_type")["probation_period"].apply(
    lambda s: s.isnull().mean())
)
print("\nInvalid dates:", df["posted_date"].isna().sum())
print("\nZero minimum salaries:", (df["salary_minimum"] == 0).sum())

print("Negative minimum salaries:", (df["salary_minimum"] < 0).sum())

print("Negative maximum salaries:", (df["salary_maximum"] < 0).sum())

print("\n========== DISTRIBUTION INVESTIGATION ==========")

print("\nCountry distribution:")
print(df["location_country"].value_counts())

print("\nCurrency distribution:")
print(df["salary_currency"].value_counts())

print("\nPlatform distribution:")
print(df["source_platform"].value_counts())

print("\nCompany count by industry:")
print(df.groupby("industry")["company_name"].nunique().sort_values(ascending=False))

print("\nCareer growth index examples:")
print(df["career_growth_index"].value_counts().head(15))

print("\nSkill count distribution:")
print(df["skill_count"].value_counts().sort_index())

print("\n========== RECORD REPETITION CHECK ==========")

print("Unique job titles:", df["job_title"].nunique())
print("Unique job descriptions:", df["job_description"].nunique())
print("Unique company names:", df["company_name"].nunique())

print("\nMost common job descriptions:")
print(df["job_description"].value_counts().head(5))

print("\nMost common company names:")
print(df["company_name"].value_counts().head(15))

print("\nSalary minimum vs maximum correlation:")
print(df[["salary_minimum", "salary_maximum"]].corr())