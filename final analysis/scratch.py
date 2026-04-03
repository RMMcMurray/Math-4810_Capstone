import pandas as pd

files = {"PUMA": "data/PUMA2.csv", "Area_Code": "data/area_code.xlsx"}

for name, path in files.items():
    try:
        if "xlsx" in path:
            df = pd.read_excel(path, nrows=0)
        else:
            df = pd.read_csv(path, nrows=0)

        print(f"--- {name} Columns ---")
        print(df.columns.tolist())
        print("\n")
    except Exception as e:
        print(f"--- Error loading {name} ---")
        print(e)
