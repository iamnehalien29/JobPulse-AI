import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/india_jobs_featured.csv")

print("Dataset shape:", df.shape)

print(df[["city_clean", "skills"]].head(10))

python_jobs = df[
    df["skills"].fillna("").str.contains(
        "python",
        case=False,
        regex=False
    )
]
python_city_counts = (
    python_jobs
    .groupby("city_clean")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\n========== TOP 15 CITIES FOR PYTHON JOBS ==========")
print(python_city_counts)

target_skills = [
    "machine learning",
    "data analysis",
    "power bi",
    "deep learning",
    "artificial intelligence"
]

skill_pattern = "|".join(target_skills)

print(skill_pattern)

df["has_data_ai_skills"] = (
    df["skills"]
    .fillna("")
    .str.contains(
        skill_pattern,
        case=False
    )
)
print(df["has_data_ai_skills"].value_counts())

data_ai_jobs = df[
    df["has_data_ai_skills"] == True
]

data_ai_city_counts = (
    data_ai_jobs
    .groupby("city_clean")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\n========== TOP 15 CITIES FOR DATA & AI JOBS ==========")
print(data_ai_city_counts)

print(
    "Total Data & AI job postings:",
    df["has_data_ai_skills"].sum()
)

# Total job postings per city
total_city_counts = (
    df
    .groupby("city_clean")
    .size()
    .sort_values(ascending=False)
)

print("\n========== TOTAL JOB POSTINGS BY CITY ==========")
print(total_city_counts.head(15))

# Combine total jobs and Data & AI jobs
city_percentage_df = pd.DataFrame({
    "total_jobs": total_city_counts,
    "data_ai_jobs": data_ai_jobs.groupby("city_clean").size()
})

city_percentage_df = city_percentage_df.fillna(0)

city_percentage_df["data_ai_percentage"] = (
    city_percentage_df["data_ai_jobs"]
    / city_percentage_df["total_jobs"]
) * 100

# Filter cities with at least 100 job postings
city_percentage_df = city_percentage_df[
    city_percentage_df["total_jobs"] >= 100
]

city_percentage_df = city_percentage_df.sort_values(
    by="data_ai_percentage",
    ascending=False
)

# Exclude non-city location categories
city_chart_df = city_percentage_df.drop(
    index=["Anywhere In India"],
    errors="ignore"
)

# Select top 10 cities by Data & AI percentage
top_cities = city_chart_df.head(10)

plt.figure(figsize=(10, 7))

plt.barh(
    top_cities.index,
    top_cities["data_ai_percentage"]
)

plt.xlabel("Data & AI Jobs (%)")
plt.ylabel("City")
plt.title("Data & AI Job Share by City")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "reports/data_ai_job_share_by_city.png",
    dpi=300,
    bbox_inches="tight"
)

print("Data & AI percentage chart saved successfully!")

city_percentage_df = city_percentage_df.sort_values(
    by="data_ai_percentage",
    ascending=False
)

print("\n========== DATA & AI JOB PERCENTAGE BY CITY ==========")
print(city_percentage_df.head(15).round(2))

city_skill_data = {}

for skill in target_skills:
    skill_jobs = df[
        df["skills"].fillna("").str.contains(
            skill,
            case=False,
            regex=False
        )
    ]
    skill_city_counts = (
        skill_jobs
        .groupby("city_clean")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    city_skill_data[skill] = skill_city_counts
comparison_df = pd.DataFrame(city_skill_data)
comparison_df = comparison_df.fillna(0)

comparison_df["total_skill_postings"] = comparison_df.sum(axis=1)

comparison_df = comparison_df.sort_values(
    by="total_skill_postings",
    ascending=False
)
print("\n========== DATA & AI SKILL DEMAND BY CITY ==========")
print(comparison_df)

plt.figure(figsize=(10, 7))

plt.barh(
    comparison_df.index,
    comparison_df["total_skill_postings"]
)

plt.xlabel("Combined Data & AI Skill Mentions")
plt.title("Data & AI Skill Demand by City")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "reports/city_skill_demand.png",
    dpi=300,
    bbox_inches="tight"
)

print("Chart saved successfully!")

skills = (
    df["skills"]
    .dropna()
    .str.split("|")
    .explode()
    .str.strip()
    .str.lower()
)

search_skills = [
    "machine learning",
    "data analysis",
    "power bi",
    "deep learning",
    "artificial intelligence"
]

print("\n========== DATA & AI SKILL COUNTS ==========")

for skill in search_skills:
    count = (skills == skill).sum()
    print(f"{skill}: {count}")

plt.show()