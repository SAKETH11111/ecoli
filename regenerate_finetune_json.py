import pandas as pd
from CodonTransformer.CodonData import prepare_training_data

# Load the validated high-CAI genes
df = pd.read_csv("data/ecoli_processed_genes.csv")

# Keep only high-CAI, drop duplicates (already unique)
df = df[df["is_high_cai"]]

# Rename columns to the names expected by CodonData
df = df.rename(
    columns={
        "dna_sequence": "dna",
        "protein_sequence": "protein",
    }
)

# Add organism column required by the model
df["organism"] = "Escherichia coli general"   # maps to id 51

# Write JSON-lines file with correct 'codons' + 'organism' keys
prepare_training_data(df, "data/finetune_set.json", shuffle=True)
print("✅  finetune_set.json regenerated")