import pandas as pd

# Load labeled dataset
df = pd.read_csv("data/protein_structure_labels.csv")

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def compute_aac(sequence):
    seq_len = len(sequence)
    return [(sequence.count(aa) / seq_len) for aa in AMINO_ACIDS]

# Compute AAC features
aac_features = df["seq"].apply(compute_aac)

# Convert to DataFrame
aac_df = pd.DataFrame(
    aac_features.tolist(),
    columns=[f"AAC_{aa}" for aa in AMINO_ACIDS]
)

# Combine with labels
df_aac = pd.concat([aac_df, df["label"]], axis=1)

print("AAC feature matrix shape:", df_aac.shape)
print(df_aac.head())

# Save ML-ready dataset
df_aac.to_csv("data/protein_aac_features.csv", index=False)
