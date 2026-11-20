
import pandas as pd

# Load featured dataset
df = pd.read_csv("data/processed/india_jobs_featured.csv")

print("Dataset shape:", df.shape)

# =================================
# 1. MONTHLY JOB POSTING ANALYSIS
# =================================

monthly_jobs = df["posted_month_name"].value_counts()

print("\n========== MONTHLY JOB POSTINGS ==========")
print(monthly_jobs)

# =================================
# 2. TOP 15 JOB ROLES
# =================================

top_roles = df["role"].value_counts().head(15)

print("\n========== TOP 15 JOB ROLES ==========")
print(top_roles)

# =================================
# 3. TOP 15 CITIES
# =================================

top_cities = df["city_clean"].value_counts().head(15)

print("\n========== TOP 15 CITIES ==========")
print(top_cities)

# =================================
# 4. TOP 15 COMPANIES
# =================================

top_companies = df["company"].value_counts().head(15)

print("\n========== TOP 15 COMPANIES ==========")
print(top_companies)
