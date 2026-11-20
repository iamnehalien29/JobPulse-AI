
import pandas as pd

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

INPUT_PATH = "data/processed/india_jobs_cleaned.csv"
OUTPUT_PATH = "data/processed/india_jobs_featured.csv"

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)


# ==========================================
# 2. DATE FEATURES
# ==========================================

df["posted_date"] = pd.to_datetime(
    df["posted_date"],
    errors="coerce"
)

df["posted_year"] = df["posted_date"].dt.year

df["posted_month"] = df["posted_date"].dt.month

df["posted_month_name"] = df["posted_date"].dt.month_name()

df["posted_quarter"] = df["posted_date"].dt.quarter

df["posted_day_name"] = df["posted_date"].dt.day_name()


# ==========================================
# 3. SKILLS FEATURES
# ==========================================

df["skills_count"] = (
    df["skills"]
    .fillna("")
    .apply(
        lambda x: len([
            skill for skill in x.split("|")
            if skill.strip()
        ])
    )
)

df["has_skills"] = df["skills_count"] > 0


# ==========================================
# 4. MISSING DATA INDICATORS
# ==========================================

df["has_salary"] = df["salary_annual_usd"].notna()

df["has_experience_data"] = df["experience_years"].notna()


# ==========================================
# 5. VALIDATION
# ==========================================

print("\n========== FEATURE ENGINEERING CHECK ==========")

print("New dataset shape:", df.shape)

print("\nNew columns:")

new_columns = [
    "posted_year",
    "posted_month",
    "posted_month_name",
    "posted_quarter",
    "posted_day_name",
    "skills_count",
    "has_skills",
    "has_salary",
    "has_experience_data"
]

print(df[new_columns].head(10).to_string(index=False))

print("\nSkills count statistics:")
print(df["skills_count"].describe())

print("\nSalary availability:")
print(df["has_salary"].value_counts())

print("\nExperience data availability:")
print(df["has_experience_data"].value_counts())


# ==========================================
# 6. SAVE FEATURED DATASET
# ==========================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nFeatured dataset saved successfully!")
print("Location:", OUTPUT_PATH)
