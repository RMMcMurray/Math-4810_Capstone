import pandas as pd
import numpy as np

# Load sales data across asset classes
files = {
    "Residential": "data/MLS_Residential_Sales.csv",
    "Multi-Family": "data/Multi-Family_MLS_Export.csv",
    "Land Sales": "data/MLS_Land_Sales.csv",
}

# Define essential variables for trending and MLR
essential_cols = [
    "List Number",
    "Parcel Number",
    "Sold Price",
    "Sold Date",
    "segment",
    "Total SqFt",
    "Year Built",
    "Lot Acres",
    "Total Bedrooms",
    "Full Baths",
    "Half Baths",
    "GLA",
    "Concessions Amount",
    "Adjusted SP",
    "Book Section",
]

# Stack dataframes and tag by segment
all_sales = pd.concat(
    [pd.read_csv(f).assign(segment=label) for label, f in files.items()],
    ignore_index=True,
)

# Neutralize missing concessions to prevent math errors
if "Concessions Amount" in all_sales.columns:
    all_sales["Concessions Amount"] = all_sales["Concessions Amount"].fillna(0)
else:
    all_sales["Concessions Amount"] = 0

# Subset to existing prioritized columns
all_sales = all_sales[[c for c in essential_cols if c in all_sales.columns]]

# Load PUMA attributes and Neighborhood codes
puma = pd.read_csv("data/PUMA2.csv")
puma.columns = puma.columns.str.strip()

area_codes = pd.read_excel(
    "data/area_code.xlsx", sheet_name=0, usecols=["ParcelId", "NbhdCode"]
)
area_codes.columns = area_codes.columns.str.strip()
area_codes = area_codes.rename(columns={"ParcelId": "Parcel Number"})

# Standardize IDs for clean joining
for df in [all_sales, puma, area_codes]:
    for col in ["Parcel Number", "List Number"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
            )

# Merge PUMA data and backfill missing physical features
df_combined = pd.merge(
    all_sales, puma, on="List Number", how="left", suffixes=("", "_puma")
)

shared_cols = [c for c in puma.columns if c in all_sales.columns and c != "List Number"]
for col in shared_cols:
    puma_col = f"{col}_puma"
    df_combined[col] = df_combined[col].combine_first(df_combined[puma_col])
    df_combined.drop(columns=[puma_col], inplace=True)

# Attach Neighborhood mapping
df_combined = pd.merge(
    df_combined,
    area_codes.drop_duplicates("Parcel Number"),
    on="Parcel Number",
    how="left",
)

# Drop incomplete records for time-series integrity
df_combined = df_combined.dropna(subset=["Sold Price", "Sold Date"])

# Categorize by price tier for subset analysis
bins = [0, 250000, 500000, 750000, 1000000, 2000000, np.inf]
labels = ["<250k", "250k-500k", "500k-750k", "750k-1M", "1M-2M", "2M+"]
df_combined["Price_Bracket"] = pd.cut(
    df_combined["Sold Price"], bins=bins, labels=labels
)

# Export modeling-ready dataset
df_combined.to_csv("data/final_data.csv", index=False)

print("Build successful: data/final_data.csv")
print(f"Final Count: {len(df_combined)} rows")
