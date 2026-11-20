import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "job_postings_2026_sample.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Reference lists (defined once) ----------
STATE_NAMES = [
    "Andhra Pradesh", "Assam", "Bihar", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Odisha",
    "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]
NON_CITY_LABELS = [
    "Any Location", "Anywhere In India", "Others",
    "Overseas", "Metros", "Middle East",
]
INTERNATIONAL_NAMES = [
    "Australia", "China", "Dubai", "Dublin", "Kenya", "Kuwait", "Oman",
    "Qatar", "Saudi Arabia", "Singapore", "South Africa",
    "United Arab Emirates", "Zimbabwe",
]
CITY_MAPPING = {
    "Bengaluru (Bangalore)": "Bengaluru",
    "Bangalore": "Bengaluru",
    "Gurgaon": "Gurugram",
    "Cochin": "Kochi",
    "Kochi (Cochin)": "Kochi",
    "Nasik": "Nashik",
    "Mysore": "Mysuru",
    "Trichy": "Tiruchirappalli",
    "Tuticorin": "Thoothukudi",
    "Pondicherry": "Puducherry",
}
DELHI_NCR = [
    "Delhi", "Delhi Ncr", "Noida", "Greater Noida",
    "Ghaziabad", "Faridabad", "Gurugram", "Gurugram Sub-District",
]
MUMBAI_METRO = ["Mumbai", "Mumbai City", "Navi Mumbai", "Thane"]

# ---------- Load ----------
df = pd.read_csv(RAW_DATA_PATH)
print("Original dataset shape:", df.shape)

india_df = df[df["country"] == "IN"].copy()

# ---------- Clean ----------
india_df["city"] = india_df["city"].str.strip().str.title()
india_df["posted_date"] = pd.to_datetime(india_df["posted_date"], errors="coerce")
# If your dates are day-first, use: format="%d-%m-%Y"

india_df["city_clean"] = india_df["city"].replace(CITY_MAPPING)

india_df["location_type"] = np.select(
    [
        india_df["city"].isna(),
        india_df["city"].isin(STATE_NAMES),
        india_df["city"].isin(NON_CITY_LABELS),
        india_df["city"].isin(INTERNATIONAL_NAMES),
    ],
    ["Missing", "State", "Flexible/Unknown", "International"],
    default="City",
)

india_df["metro_region"] = np.select(
    [
        india_df["city_clean"].isin(DELHI_NCR),
        india_df["city_clean"].isin(MUMBAI_METRO),
        india_df["city_clean"] == "Bengaluru",
    ],
    ["Delhi NCR", "Mumbai Metropolitan Region", "Bengaluru"],
    default="Other / Unclassified",
)

# ---------- Checks ----------
print("\n========== CLEANING CHECK ==========")
print("India dataset shape:", india_df.shape)
print("Date range:", india_df["posted_date"].min(), "to", india_df["posted_date"].max())
print("Invalid dates:", india_df["posted_date"].isna().sum())

print("\nMissing values (India):")
missing = pd.DataFrame({
    "Missing_Count": india_df.isnull().sum(),
    "Missing_Percentage": (india_df.isnull().mean() * 100).round(2),
}).sort_values("Missing_Percentage", ascending=False)
print(missing)

print("\n========== DUPLICATES ==========")
print("Duplicate posting IDs:", india_df["posting_id"].duplicated().sum())
dup_cols = ["title", "company", "city"]
print("Same title/company/city:", india_df.duplicated(subset=dup_cols, keep=False).sum())
print("Same title/company/city/date:",
      india_df.duplicated(subset=dup_cols + ["posted_date"], keep=False).sum())

print("\n========== LOCATION QUALITY ==========")
print("Unique cities before cleaning:", india_df["city"].nunique())
print("Unique cities after cleaning:", india_df["city_clean"].nunique())
print("\nLocation types (%):")
print(india_df["location_type"].value_counts(normalize=True).mul(100).round(2))
print("\nMetro regions (%):")
print(india_df["metro_region"].value_counts(normalize=True).mul(100).round(2))

print("\nTop 20 cleaned cities:")
print(india_df["city_clean"].value_counts().head(20))

# ---------- Save ----------
OUTPUT_PATH = PROCESSED_DIR / "india_jobs_cleaned.csv"
india_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned data to {OUTPUT_PATH}")