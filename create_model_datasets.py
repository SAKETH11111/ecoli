import pandas as pd
import json
import os
from CodonTransformer.CodonData import prepare_training_data

def main():
    """
    Main function to partition the processed data into fine-tuning and test sets.
    """
    # Ensure the data directory exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found. Please run prepare_ecoli_data.py first.")
        return

    # Load the processed data
    processed_data_path = 'data/ecoli_processed_genes.csv'
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        return
        
    df_processed = pd.read_csv(processed_data_path)

    # Create the fine-tuning set
    df_finetune = df_processed[df_processed['is_high_cai'] == True].copy()
    df_finetune.drop_duplicates(subset=['dna_sequence'], inplace=True)
    df_finetune.rename(columns={'dna_sequence': 'dna', 'protein_sequence': 'protein'}, inplace=True)
    df_finetune['organism'] = "Escherichia coli general"
    
    finetune_output_path = 'data/finetune_set.json'
    prepare_training_data(df_finetune, finetune_output_path, shuffle=True)
    print(f"Fine-tuning set saved to {finetune_output_path} with {len(df_finetune)} records.")

    # Create the test set
    df_test_pool = df_processed[df_processed['is_high_cai'] == False].copy()
    df_test = df_test_pool.sample(n=100, random_state=42) # for reproducibility
    df_test['organism'] = 51 # E. coli general
    df_test.rename(columns={'dna_sequence': 'codons'}, inplace=True)
    test_records = df_test[['codons', 'organism']].to_dict(orient='records')

    test_output_path = 'data/test_set.json'
    with open(test_output_path, 'w') as f:
        json.dump(test_records, f, indent=4)
    print(f"Test set saved to {test_output_path} with {len(df_test)} records.")

if __name__ == "__main__":
    main()