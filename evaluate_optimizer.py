import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm

from CodonTransformer.CodonData import (
    download_codon_frequencies_from_kazusa,
    get_codon_frequencies,
)
from CodonTransformer.CodonPrediction import (
    load_model,
    predict_dna_sequence,
    get_high_frequency_choice_sequence_optimized,
)
from CodonTransformer.CodonEvaluation import (
    calculate_dtw_distance,
    calculate_homopolymer_runs,
    calculate_tAI,
    count_negative_cis_elements,
    get_GC_content,
    get_ecoli_tai_weights,
    get_min_max_profile,
    get_sequence_similarity,
    scan_for_restriction_sites,
    calculate_ENC,
    calculate_CPB,
    calculate_SCUO,
)
from CodonTransformer.CodonUtils import DNASequencePrediction

def translate_dna_to_protein(dna_sequence: str) -> str:
    """Translate DNA sequence to protein sequence."""
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    
    protein = ""
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3].upper()
        if len(codon) == 3:
            aa = codon_table.get(codon, 'X')
            if aa == '*':  # Stop codon
                break
            protein += aa
    
    return protein

def main(args):
    """Main function to run the evaluation."""
    # --- 1. Setup ---
    print("--- Phase 4: Evaluation & Benchmarking ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Robust JSON/JSONL loading ---
    with open("data/test_set.json", "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            test_set = json.load(f)
        else:
            test_set = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(test_set)} proteins from the test set.")

    # --- 2. Initialize Models ---
    print("Initializing models...")
    # Fine-tuned model
    finetuned_model_path = args.checkpoint_path
    finetuned_model = load_model(model_path=finetuned_model_path, device=device)
    print(f"Fine-tuned model loaded from {finetuned_model_path}.")

    # Base model
    base_model = load_model(device=device)
    print("Base model loaded from Hugging Face.")

    # --- 3. Prepare Evaluation Utilities ---
    print("Preparing evaluation utilities...")
    # For CAI, use a broad E. coli reference set.
    natural_csv = "data/ecoli_processed_genes.csv"
    natural_df = pd.read_csv(natural_csv)
    ref_sequences = natural_df['dna_sequence'].tolist()
    cai_weights = relative_adaptiveness(sequences=ref_sequences)
    print("CAI weights generated from the natural E. coli gene set.")

    tai_weights = get_ecoli_tai_weights()
    print("tAI weights for E. coli loaded.")

    try:
        codon_frequencies = download_codon_frequencies_from_kazusa(taxonomy_id=83333)
        print("Codon frequencies for E. coli loaded from Kazusa.")
    except Exception as e:
        warnings.warn(
            f"Kazusa codon frequency download failed ({e}). Computing frequencies from finetune set."
        )
        dna_sequences = [dna for dna in ref_sequences]  # already validated
        codon_frequencies = get_codon_frequencies(
            dna_sequences, organism="Escherichia coli general"
        )

    # Generate a reference %MinMax profile from the natural sequences
    # Use the natural_df for DTW reference as well
    natural_dna_for_dtw = natural_df['dna_sequence'].tolist()
    reference_profiles = [
        get_min_max_profile(seq, codon_frequencies) for seq in natural_dna_for_dtw
    ]
    # Average the profiles to create a single reference profile
    valid_profiles = [p for p in reference_profiles if p and not all(v is None for v in p)]
    if not valid_profiles:
        raise ValueError("No valid reference profiles found for DTW calculation.")

    max_len = max(len(p) for p in valid_profiles)
    padded_profiles = [
        np.pad(
            np.array([v for v in p if v is not None]),
            (0, max_len - len([v for v in p if v is not None])),
            "constant",
            constant_values=np.nan,
        )
        for p in valid_profiles
    ]
    avg_reference_profile = np.nanmean(padded_profiles, axis=0)
    print("Reference %MinMax profile generated from natural E. coli gene set.")

    # --- 4. Run Evaluation Loop ---
    results = []
    print("Starting evaluation loop...")
    for item in tqdm(test_set, desc="Evaluating Proteins"):
        if "protein_sequence" in item:
            protein_sequence = item["protein_sequence"]
        else:
            # Translate DNA sequence to protein sequence
            dna_sequence = item["codons"]
            protein_sequence = translate_dna_to_protein(dna_sequence)

        # --- Generate Sequences ---
        # A. Fine-tuned Model
        try:
            output = predict_dna_sequence(
                protein=protein_sequence,
                organism="Escherichia coli general",
                device=device,
                model=finetuned_model,
                deterministic=True,
                match_protein=True,
                use_constrained_search=args.use_constrained_search,
                gc_bounds=tuple(args.gc_bounds),
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                diversity_penalty=args.diversity_penalty,
            )
        except ValueError as e:
            if "Beam rescue failed" in str(e):
                print(f"Warning: Constrained search failed for protein {protein_sequence[:20]}... Falling back to standard generation.")
                # Fallback to standard generation without constraints
                output = predict_dna_sequence(
                    protein=protein_sequence,
                    organism="Escherichia coli general",
                    device=device,
                    model=finetuned_model,
                    deterministic=True,
                    match_protein=True,
                    use_constrained_search=False,
                )
            else:
                raise e
        if isinstance(output, list):
            finetuned_dna = output[0].predicted_dna
        else:
            finetuned_dna = output.predicted_dna

        # B. Base Model
        base_output = predict_dna_sequence(
            protein=protein_sequence,
            organism="Escherichia coli general",
            device=device,
            model=base_model,
            deterministic=True,
            match_protein=True,
        )
        if isinstance(base_output, list):
            base_dna = base_output[0].predicted_dna
        else:
            base_dna = base_output.predicted_dna

        # C. Naive High-Frequency Baseline
        naive_dna = get_high_frequency_choice_sequence_optimized(
            protein=protein_sequence, codon_frequencies=codon_frequencies
        )

        sequences_to_evaluate = {
            "fine_tuned": finetuned_dna,
            "base": base_dna,
            "naive_hfc": naive_dna,
        }

        # --- Calculate Metrics for each sequence ---
        for model_name, dna_sequence in sequences_to_evaluate.items():
            # Primary Metrics
            cai_score = CAI(dna_sequence, weights=cai_weights)
            gc_content = get_GC_content(dna_sequence)
            tai_score = calculate_tAI(dna_sequence, tai_weights)

            # Health Metrics
            restriction_sites = scan_for_restriction_sites(dna_sequence)
            neg_cis_elements = count_negative_cis_elements(dna_sequence)
            homopolymer_runs = calculate_homopolymer_runs(dna_sequence)

            # Advanced Metrics
            min_max_profile = get_min_max_profile(dna_sequence, codon_frequencies)
            if not min_max_profile or all(v is None for v in min_max_profile):
                dtw_distance = np.inf
            else:
                dtw_distance = calculate_dtw_distance(
                    min_max_profile, avg_reference_profile
                )
            
            # Enhanced Codon Usage Metrics
            enc_value = calculate_ENC(dna_sequence)
            cpb_value = calculate_CPB(dna_sequence, ref_sequences[:100])  # Use subset for efficiency
            scuo_value = calculate_SCUO(dna_sequence)

            # Store results
            results.append(
                {
                    "protein": protein_sequence,
                    "model": model_name,
                    "dna_sequence": dna_sequence,
                    "cai": cai_score,
                    "tai": tai_score,
                    "gc_content": gc_content,
                    "restriction_sites": restriction_sites,
                    "neg_cis_elements": neg_cis_elements,
                    "homopolymer_runs": homopolymer_runs,
                    "dtw_distance": dtw_distance,
                    "enc": enc_value,
                    "cpb": cpb_value,
                    "scuo": scuo_value,
                }
            )

    # --- 5. Save Results ---
    print("Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"Evaluation complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the CodonTransformer model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/ecoli-codon-optimizer/finetune_best.ckpt",
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluation_results.csv",
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--use_constrained_search",
        action="store_true",
        help="Use constrained beam search with GC bounds",
    )
    parser.add_argument(
        "--gc_bounds",
        type=float,
        nargs=2,
        default=[0.50, 0.54],
        help="GC content bounds for constrained search (min max)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Beam size for constrained search",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.2,
        help="Length penalty for beam search scoring",
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=0.1,
        help="Diversity penalty to reduce repetitive sequences",
    )
    args = parser.parse_args()
    main(args)