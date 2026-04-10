import pandas as pd

# Define the file paths
files = {
    "Residential": "data/MLS_Residential_Sales.csv",
    "Multi-Family": "data/Multi-Family_MLS_Export.csv",
    "Land Sales": "data/MLS_Land_Sales.csv",
    "PUMA": "data/final_puma.csv",
}

print("--- CSV Column Manifest ---")

for label, path in files.items():
    try:
        # Load only the header row to be fast
        df = pd.read_csv(path, nrows=0)

        print(f"\n{label.upper()} ({path})")
        print(f"Total Columns: {len(df.columns)}")
        print(list(df.columns))

    except Exception as e:
        print(f"\nCould not read {label}: {e}")

print("\n--- End of Manifest ---")
