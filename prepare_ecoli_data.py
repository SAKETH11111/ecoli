import pandas as pd
from Bio.Seq import Seq
import os

def is_valid_sequence(dna_seq: str) -> bool:
    """
    Applies a series of validation checks to a DNA sequence.

    Args:
        dna_seq (str): The DNA sequence to validate.

    Returns:
        bool: True if the sequence is valid, False otherwise.
    """
    # Structural Integrity Checks
    if len(dna_seq) % 3 != 0:
        return False
    if not dna_seq.upper().startswith(('ATG', 'TTG', 'CTG', 'GTG')):
        return False
    if not dna_seq.upper().endswith(('TAA', 'TAG', 'TGA')):
        return False
    
    # Check for internal stop codons
    codons = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 3, 3)]
    if any(codon in ['TAA', 'TAG', 'TGA'] for codon in codons):
        return False

    # Compositional Purity Checks
    if not all(c in 'ATGC' for c in dna_seq.upper()):
        return False

    return True

def main():
    """
    Main function to process and validate E. coli gene data.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load datasets
    print("Loading data from CSV files...")
    df_all = pd.read_csv("data/CAI.csv", header=0, names=['gene_id', 'cai_score', 'drop1', 'drop2', 'dna_sequence', 'drop3'])
    df_high_cai = pd.read_csv("data/Database 3_4300 gene.csv", header=0, names=['dna_sequence'])


    # Use the DNA sequence as the identifier for high CAI genes
    high_cai_sequences = set(df_high_cai['dna_sequence'])

    validated_genes = []
    for index, row in df_all.iterrows():
        gene_id = row['gene_id']
        dna_sequence = str(row['dna_sequence'])

        if is_valid_sequence(dna_sequence):
            protein_sequence = str(Seq(dna_sequence).translate())
            is_high_cai = dna_sequence in high_cai_sequences
            
            validated_genes.append({
                'gene_id': gene_id,
                'dna_sequence': dna_sequence,
                'protein_sequence': protein_sequence,
                'cai_score': row.get('cai_score', None),
                'is_high_cai': is_high_cai
            })

    df_processed = pd.DataFrame(validated_genes)
    
    # Save the processed data
    output_path = 'data/ecoli_processed_genes.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Total validated genes: {len(df_processed)}")

if __name__ == "__main__":
    main()