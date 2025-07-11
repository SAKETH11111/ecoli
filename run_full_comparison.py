#!/usr/bin/env python3
"""
Full Comparison Evaluation Script
Compares three methods:
1. Base CodonTransformer model (from HuggingFace)
2. Fine-tuned model (your best performing checkpoint)
3. Naive High Frequency Choice (HFC) baseline
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple, Any

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


def calculate_comprehensive_metrics(
    dna_sequence: str,
    protein_sequence: str,
    cai_weights: Dict[str, float],
    tai_weights: Dict[str, float],
    codon_frequencies: Dict,
    reference_profile: List[float],
    ref_sequences: List[str],
) -> Dict[str, float]:
    """Calculate comprehensive metrics for a DNA sequence."""
    if not dna_sequence:
        return {
            'cai': 0.0,
            'tai': 0.0,
            'gc_content': 0.0,
            'restriction_sites': float('inf'),
            'neg_cis_elements': float('inf'),
            'homopolymer_runs': float('inf'),
            'dtw_distance': float('inf'),
            'enc': 0.0,
            'cpb': 0.0,
            'scuo': 0.0,
        }
    
    # Basic metrics
    cai_score = CAI(dna_sequence, weights=cai_weights)
    tai_score = calculate_tAI(dna_sequence, tai_weights)
    gc_content = get_GC_content(dna_sequence)
    restriction_sites = scan_for_restriction_sites(dna_sequence)
    neg_cis_elements = count_negative_cis_elements(dna_sequence)
    homopolymer_runs = calculate_homopolymer_runs(dna_sequence)
    
    # DTW distance
    try:
        min_max_profile = get_min_max_profile(dna_sequence, codon_frequencies)
        if reference_profile and min_max_profile:
            dtw_distance = calculate_dtw_distance(min_max_profile, reference_profile)
        else:
            dtw_distance = 0.0
    except Exception:
        dtw_distance = 0.0
    
    # Enhanced codon usage metrics
    try:
        enc_value = calculate_ENC(dna_sequence)
    except Exception:
        enc_value = 45.0  # Default E. coli value
    
    try:
        cpb_value = calculate_CPB(dna_sequence, ref_sequences[:100])
    except Exception:
        cpb_value = 0.0
    
    try:
        scuo_value = calculate_SCUO(dna_sequence)
    except Exception:
        scuo_value = 0.5
    
    return {
        'cai': cai_score,
        'tai': tai_score,
        'gc_content': gc_content,
        'restriction_sites': restriction_sites,
        'neg_cis_elements': neg_cis_elements,
        'homopolymer_runs': homopolymer_runs,
        'dtw_distance': dtw_distance,
        'enc': enc_value,
        'cpb': cpb_value,
        'scuo': scuo_value,
    }


def main(args):
    """Main function to run the full comparison evaluation."""
    print("=== Full Comparison Evaluation ===")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    with open(args.test_data_path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            test_set = json.load(f)
        else:
            test_set = [json.loads(line) for line in f if line.strip()]
    
    # Limit test set size if requested
    if args.max_test_proteins > 0:
        test_set = test_set[:args.max_test_proteins]
    
    print(f"Loaded {len(test_set)} proteins from the test set.")
    
    # Load models
    print("Loading models...")
    
    # Load base model (from HuggingFace)
    base_model = load_model(device=device)
    print("Base model loaded from HuggingFace")
    
    # Load fine-tuned model
    finetuned_model = load_model(model_path=args.checkpoint_path, device=device)
    print(f"Fine-tuned model loaded from {args.checkpoint_path}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    
    # Prepare evaluation utilities
    print("Preparing evaluation utilities...")
    
    # CAI weights
    natural_csv = args.natural_sequences_path
    natural_df = pd.read_csv(natural_csv)
    ref_sequences = natural_df['dna_sequence'].tolist()
    cai_weights = relative_adaptiveness(sequences=ref_sequences)
    print("CAI weights generated")
    
    # tAI weights
    tai_weights = get_ecoli_tai_weights()
    print("tAI weights loaded")
    
    # Codon frequencies
    try:
        codon_frequencies = download_codon_frequencies_from_kazusa(taxonomy_id=83333)
        print("Codon frequencies loaded from Kazusa")
    except Exception as e:
        print(f"Warning: Kazusa download failed ({e}). Using local frequencies.")
        codon_frequencies = get_codon_frequencies(
            ref_sequences, organism="Escherichia coli general"
        )
    
    # Reference profile for DTW
    reference_profiles = [
        get_min_max_profile(seq, codon_frequencies) for seq in ref_sequences[:100]
    ]
    valid_profiles = [p for p in reference_profiles if p and not all(v is None for v in p)]
    
    if valid_profiles:
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
        avg_reference_profile = np.nanmean(padded_profiles, axis=0).tolist()
    else:
        avg_reference_profile = []
    
    print("Reference profile generated")
    
    # Run evaluation
    all_results = []
    
    print("Starting full comparison evaluation...")
    for i, item in enumerate(tqdm(test_set, desc="Evaluating proteins")):
        # Get protein sequence
        if "protein_sequence" in item:
            protein_sequence = item["protein_sequence"]
        else:
            dna_sequence = item["codons"]
            protein_sequence = translate_dna_to_protein(dna_sequence)
        
        # Skip if protein is too short or too long
        if len(protein_sequence) < 10 or len(protein_sequence) > 1000:
            continue
        
        # 1. Base model prediction
        try:
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
                
            base_metrics = calculate_comprehensive_metrics(
                dna_sequence=base_dna,
                protein_sequence=protein_sequence,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=avg_reference_profile,
                ref_sequences=ref_sequences,
            )
            
            all_results.append({
                'protein_id': i,
                'protein_sequence': protein_sequence,
                'protein_length': len(protein_sequence),
                'method': 'base_model',
                'dna_sequence': base_dna,
                'dna_length': len(base_dna),
                **base_metrics,
            })
            
        except Exception as e:
            print(f"Warning: Base model generation failed for protein {i}: {str(e)}")
        
        # 2. Fine-tuned model prediction
        try:
            finetuned_output = predict_dna_sequence(
                protein=protein_sequence,
                organism="Escherichia coli general",
                device=device,
                model=finetuned_model,
                deterministic=True,
                match_protein=True,
            )
            
            if isinstance(finetuned_output, list):
                finetuned_dna = finetuned_output[0].predicted_dna
            else:
                finetuned_dna = finetuned_output.predicted_dna
                
            finetuned_metrics = calculate_comprehensive_metrics(
                dna_sequence=finetuned_dna,
                protein_sequence=protein_sequence,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=avg_reference_profile,
                ref_sequences=ref_sequences,
            )
            
            all_results.append({
                'protein_id': i,
                'protein_sequence': protein_sequence,
                'protein_length': len(protein_sequence),
                'method': 'fine_tuned_original',
                'dna_sequence': finetuned_dna,
                'dna_length': len(finetuned_dna),
                **finetuned_metrics,
            })
            
        except Exception as e:
            print(f"Warning: Fine-tuned model generation failed for protein {i}: {str(e)}")
        
        # 3. Naive HFC baseline
        try:
            naive_dna = get_high_frequency_choice_sequence_optimized(
                protein=protein_sequence, codon_frequencies=codon_frequencies
            )
            
            naive_metrics = calculate_comprehensive_metrics(
                dna_sequence=naive_dna,
                protein_sequence=protein_sequence,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=avg_reference_profile,
                ref_sequences=ref_sequences,
            )
            
            all_results.append({
                'protein_id': i,
                'protein_sequence': protein_sequence,
                'protein_length': len(protein_sequence),
                'method': 'naive_hfc',
                'dna_sequence': naive_dna,
                'dna_length': len(naive_dna),
                **naive_metrics,
            })
            
        except Exception as e:
            print(f"Warning: Naive HFC generation failed for protein {i}: {str(e)}")
    
    # Save results
    print("Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_path, index=False)
    
    # Print summary
    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Total proteins evaluated: {len(test_set)}")
    print(f"Total sequences generated: {len(all_results)}")
    print(f"Results saved to: {args.output_path}")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    summary = results_df.groupby('method')[['cai', 'tai', 'gc_content', 'restriction_sites', 'neg_cis_elements', 'homopolymer_runs']].mean().round(4)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Comparison Evaluation")
    
    # Input/Output paths
    parser.add_argument("--checkpoint_path", type=str, default="models/alm-enhanced-training/balanced_alm_finetune.ckpt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--test_data_path", type=str, default="data/test_set.json",
                        help="Path to test dataset")
    parser.add_argument("--natural_sequences_path", type=str, default="data/ecoli_processed_genes.csv",
                        help="Path to natural E. coli sequences for CAI calculation")
    parser.add_argument("--output_path", type=str, default="results/evaluation_results.csv",
                        help="Path to save evaluation results")
    
    # Model parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
    # Evaluation parameters
    parser.add_argument("--max_test_proteins", type=int, default=0,
                        help="Maximum number of proteins to test (0 for all)")
    
    args = parser.parse_args()
    main(args) 