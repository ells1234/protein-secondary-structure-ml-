import pandas as pd

df = pd.read_csv("data/2022-12-17-pdb-intersect-pisces_pc30_r2.5.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n--- Sample row (row 0) ---")
row = df.iloc[0]

for col in df.columns:
    val = row[col]
    if isinstance(val, str) and len(val) > 120:
        val = val[:120] + "..."
    print(f"{col}: {val}")
