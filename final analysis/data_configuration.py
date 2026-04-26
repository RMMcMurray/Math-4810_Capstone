import pandas as pd
import numpy as np
import pwlf
import statsmodels.formula.api as smf

# --- DATA ACQUISITION & INITIAL CLEANING ---

puma = pd.read_csv("data/final_puma.csv")
puma.columns = puma.columns.str.strip()

impute_zero_cols = [
    "FullBaths_Total",
    "HalfBaths_Total",
    "BsmtFinishPct_Weighted",
    "Laundry_Count",
    "KitchenCount",
    "CarportArea",
    "CarportCapacity",
    "GarageArea",
    "GarageCapacity",
    "Tot Bsmt",
    "SqFt_DetGarage",
    "Count_ DetCarport",
    "SqFt_DetCarport",
    "Count_Barn",
    "Sqft_Barn",
    "Count_Guest_House",
    "SqFt_Guest_House",
    "Count_Pools",
    "SqFt_Pools",
    "Count_Recreational_Courts",
    "SqFt_Recreational_Courts",
    "Count_Shed",
    "SqFt_Shed",
    "Count_Gazebo_Pavilion",
    "SqFt_Gazebo_Pavilion",
    "Count_DetGarage",
]

for col in impute_zero_cols:
    if col in puma.columns:
        puma[col] = puma[col].fillna(0)

files = {
    "Residential": "data/MLS_Residential_Sales.csv",
    "Multi-Family": "data/Multi-Family_MLS_Export.csv",
    "Land Sales": "data/MLS_Land_Sales.csv",
}

mls_mapping = {
    "Parcel Number": "Parcel Number",
    "Sold Price": "Sold Price",
    "Sold Date": "Sold Date",
    "Concessions Amount": "Concessions Amount",
    "Total SqFt": "TotGLA",
    "Finished SqFt": "TotGLA",
    "Lot Acres": "Acreage",
    "Year Built": "Main_YearBuilt",
    "Subdivision": "Subdivision",
}

all_sales_list = []
for label, path in files.items():
    temp_df = pd.read_csv(path).assign(segment=label)
    cols_to_use = [c for c in mls_mapping.keys() if c in temp_df.columns]
    temp_df = temp_df[cols_to_use + ["segment", "List Number"]]
    temp_df = temp_df.rename(columns=mls_mapping)
    all_sales_list.append(temp_df)

all_sales = pd.concat(all_sales_list, ignore_index=True)

# --- DATABASE INTEGRATION & FEATURE ENGINEERING ---

for df in [puma, all_sales]:
    id_col = "ParcelId" if "ParcelId" in df.columns else "Parcel Number"
    df[id_col] = (
        df[id_col].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
    )

puma = puma.rename(columns={"ParcelId": "Parcel Number"})
df_combined = pd.merge(
    puma, all_sales, on="Parcel Number", how="left", suffixes=("", "_mls")
)

# Feature: Truncate Neighborhood Code to the first letter
if "NbhdCode2" in df_combined.columns:
    df_combined["NbhdCode2"] = df_combined["NbhdCode2"].astype(str).str[0]

# Feature: Convert Year Built to Age
current_year = 2026
if "Main_YearBuilt" in df_combined.columns:
    df_combined["Age"] = current_year - df_combined["Main_YearBuilt"]

for col in ["TotGLA", "Acreage", "Main_YearBuilt", "Subdivision"]:
    mls_col = f"{col}_mls"
    if mls_col in df_combined.columns:
        df_combined[col] = df_combined[col].combine_first(df_combined[mls_col])
        df_combined.drop(columns=[mls_col], inplace=True)

df_combined["Concessions Amount"] = df_combined["Concessions Amount"].fillna(0)
df_combined["Adj_Price"] = df_combined["Sold Price"] - df_combined["Concessions Amount"]
df_combined["Sold Date"] = pd.to_datetime(df_combined["Sold Date"], errors="coerce")

cols_to_drop = [
    "MainImp.GroupCode",
    "Res Imp Count",
    "ImpCode",
    "Area",
    "ReviewedBy",
    "ReviewedDate",
    "Lng",
    "Lat",
    "Jurisdiction",
    "Subdivision",
    "PropType",
    "IsIncluded",
    "ImpOnly",
    "Active",
    "TaxYear",
    "RecordType",
    "SpecificPropType",
    "Main_YearBuilt",
    "Age"
]

# --- MARKET STABILIZATION & OUTLIER HANDLING ---

df_adj = df_combined.dropna(subset=["Sold Date", "Sold Price"]).copy()

# Simple Outlier Filter: Remove top/bottom 1% to prevent regression skew
p_low = df_adj["Adj_Price"].quantile(0.01)
p_high = df_adj["Adj_Price"].quantile(0.99)
df_adj = df_adj[(df_adj["Adj_Price"] > p_low) & (df_adj["Adj_Price"] < p_high)]

sqft_col = "TotGLA" if "TotGLA" in df_adj.columns else "Total SqFt"
year_col = "Main_YearBuilt" if "Main_YearBuilt" in df_adj.columns else "Year Built"

df_adj["ln_Price"] = np.log(df_adj["Adj_Price"])
df_adj["ln_GLA"] = np.log(df_adj[sqft_col].replace(0, np.nan))

start_date = pd.Timestamp("2021-01-01")
x_days = (df_adj["Sold Date"] - start_date).dt.days.values
pw_model = pwlf.PiecewiseLinFit(x_days, df_adj["ln_Price"].values)
knots_days = pw_model.fit(3)
knot_dates = [start_date + pd.Timedelta(days=int(k)) for k in knots_days]

for i in range(len(knot_dates) - 1):
    col_name = f"Time_{i + 1}"
    start, end = knot_dates[i], knot_dates[i + 1]
    df_adj[col_name] = df_adj["Sold Date"].apply(
        lambda x: max(0, (min(x, end) - start).days / 30.44) if pd.notna(x) else 0
    )

df_adj["Trended_Price"] = df_adj["Adj_Price"].copy()
time_cols = [c for c in df_adj.columns if c.startswith("Time_")]

for seg in ["Residential", "Multi-Family", "Land Sales"]:
    df_seg = df_adj[df_adj["segment"] == seg].copy()
    if len(df_seg) < 20:
        continue

    valid_buckets = [col for col in time_cols if df_seg[col].max() > 0]

    if seg != "Land Sales":
        formula = f"ln_Price ~ {' + '.join(valid_buckets)} + ln_GLA + Q('{year_col}')"
        df_clean = df_seg.dropna(subset=["ln_GLA", year_col])
    else:
        formula = f"ln_Price ~ {' + '.join(valid_buckets)}"
        df_clean = df_seg

    try:
        res_ols = smf.ols(formula, data=df_clean).fit()
        adj_log = 0
        for idx, col in enumerate(time_cols):
            if col in res_ols.params:
                bucket_cap = (knot_dates[idx + 1] - knot_dates[idx]).days / 30.44
                mask = df_adj["segment"] == seg
                remaining = bucket_cap - df_adj.loc[mask, col]
                adj_log += res_ols.params[col] * np.maximum(0, remaining)

        df_adj.loc[df_adj["segment"] == seg, "Trended_Price"] *= np.exp(adj_log)
    except Exception as e:
        print(f"Regression failed for {seg}: {e}")

# Final drop and export
df_final_raw = df_combined.dropna(subset=["Sold Date", "Sold Price"])
df_final_raw = df_final_raw.drop(
    columns=[c for c in cols_to_drop if c in df_final_raw.columns]
)
df_final_raw.to_csv("data/final_data.csv", index=False)

df_adj = df_adj.drop(columns=[c for c in cols_to_drop if c in df_adj.columns])
df_adj.to_csv("data/final_adjusted.csv", index=False)
print("FINISHED")