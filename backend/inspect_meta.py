import pandas as pd, os, sys

p = "styles.csv"
print("Looking for", p, "in", os.getcwd())

if not os.path.exists(p):
    print("ERROR: file not found:", p)
    sys.exit(2)

try:
    df = pd.read_csv(p)
except Exception as e:
    print("ERROR reading", p, ":", e)
    sys.exit(3)

print("Loaded", p, "rows =", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head(5).to_dict(orient="records"))

if "season" in df.columns:
    print("\nUnique season values:", sorted(
        df["season"].dropna().astype(str).str.strip().unique().tolist()
    ))
else:
    print("\nNo 'season' column found.")

if "subCategory" in df.columns or "sub_category" in df.columns:
    col = "subCategory" if "subCategory" in df.columns else "sub_category"
    print("\nTop subCategory values:")
    print(
        df[col].fillna("")
        .astype(str).str.strip().str.lower()
        .value_counts().head(20)
    )
else:
    print("\nNo subCategory column found.")
