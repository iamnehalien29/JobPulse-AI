
import pandas as pd

df = pd.read_csv("data/processed/india_jobs_featured.csv")

df["posted_date"] = pd.to_datetime(df["posted_date"])

# TASK A: Monthly coverage

df["month_number"] = df["posted_date"].dt.month

monthly_coverage = df.groupby("month_number").agg(
    total_postings=("posted_date", "count"),
    unique_posting_dates=("posted_date", "nunique")
)

monthly_coverage["avg_postings_per_day"] = (
    monthly_coverage["total_postings"] /
    monthly_coverage["unique_posting_dates"]
)

print("\n========== MONTHLY COVERAGE ==========")
print(monthly_coverage)


# TASK B: Missing categorical information

categorical_columns = [
    "role",
    "seniority",
    "category",
    "city_clean",
    "region"
]

missing_info = df[categorical_columns].isnull().sum()

missing_percentages = (missing_info / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing_Count": missing_info,
    "Missing_Percentage": missing_percentages
})

print("\n========== MISSING DATA ==========")
print(missing_summary.sort_values(
    by="Missing_Percentage",
    ascending=False
))


# TASK C: Company concentration

top_10 = df["company"].value_counts().head(10)

combined_count = top_10.sum()

percentage = (
    combined_count / df["company"].count()
) * 100

unique_companies = df["company"].nunique()

print("\n========== COMPANY CONCENTRATION ==========")
print("Top 10 companies:")
print(top_10)

print("Combined postings:", combined_count)
print("Percentage:", round(percentage, 2), "%")
print("Unique companies:", unique_companies)
