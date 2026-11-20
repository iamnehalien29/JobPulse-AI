import pandas as pd

df = pd.read_csv("data/processed/india_jobs_featured.csv")

print("Dataset shape:", df.shape)

skills = df["skills"].dropna()
skills = skills.str.split("|", regex=False).explode()
skills = skills.str.strip().str.lower()

skills = skills[skills.ne("")]

top_skills = skills.value_counts().head(25)

print(top_skills)

skill_categories = {
    "Programming": [
        "python",
        "java",
        "javascript",
        "typescript",
        "css"
    ],
    "Cloud & DevOps": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes"
    ],
    "Data & AI": [
        "sql",
        "data analysis",
        "machine learning",
        "power bi"
    ],
    "Business & Soft Skills": [
    "communication skills",
    "communication",
    "negotiation",
    "project management",
    "stakeholder management",
    "business development",
    "sales"
]
    
}
category_job_counts = {}

for category, skill_list in skill_categories.items():

    job_matches = df["skills"].fillna("").apply(
        lambda job_skills: any(
            skill in [
                item.strip().lower()
                for item in job_skills.split("|")
            ]
            for skill in skill_list
        )
    )

    category_job_counts[category] = job_matches.sum()


print("\n========== UNIQUE JOB POSTINGS BY CATEGORY ==========")

for category, count in category_job_counts.items():
    print(category, ":", count)


category_counts = {}

print("\n========== JOB CATEGORY COVERAGE ==========")

for category, count in category_job_counts.items():
    percentage = (count / len(df)) * 100
    print(category, ":", round(percentage, 2), "%")

for category, skill_list in skill_categories.items():
    count = skills[skills.isin(skill_list)].count()
    category_counts[category] = count

print("\n========== SKILL CATEGORY DEMAND ==========")

for category, count in category_counts.items():
    print(category, ":", count)

# Calculate total skill mentions
total_mentions = sum(category_counts.values())

print("\n========== SKILL CATEGORY PERCENTAGES ==========")

for category, count in category_counts.items():
    percentage = (count / total_mentions) * 100
    print(category, ":", round(percentage, 2), "%")