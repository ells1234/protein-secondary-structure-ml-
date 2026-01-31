import pandas as pd

# Load dataset
df = pd.read_csv("data/2022-12-17-pdb-intersect-pisces_pc30_r2.5.csv")

# Optional: remove proteins with non-standard amino acids
df = df[df["has_nonstd_aa"] == False].reset_index(drop=True)

def compute_structure_props(sst3):
    length = len(sst3)
    h = sst3.count("H") / length
    e = sst3.count("E") / length
    c = sst3.count("C") / length
    return pd.Series([length, h, e, c])

# Compute length + structure proportions
df[["length", "helix_pct", "sheet_pct", "coil_pct"]] = df["sst3"].apply(compute_structure_props)

def dominant_label(row):
    if row["helix_pct"] >= row["sheet_pct"] and row["helix_pct"] >= row["coil_pct"]:
        return "Helix"
    elif row["sheet_pct"] >= row["coil_pct"]:
        return "Sheet"
    else:
        return "Coil"

# Assign class label
df["label"] = df.apply(dominant_label, axis=1)

# Keep only what we need for ML
df_labels = df[["pdb_id", "seq", "length", "helix_pct", "sheet_pct", "coil_pct", "label"]]

print("Processed dataset shape:", df_labels.shape)
print(df_labels.head())

# Save for next step
df_labels.to_csv("data/protein_structure_labels.csv", index=False)
