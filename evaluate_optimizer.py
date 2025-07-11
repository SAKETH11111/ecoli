import sys
sys.path.insert(0, '/home/saketh/ecoli/codon_env/lib/python3.11/site-packages')
"""
Enhanced Evaluation Script for CodonTransformer
-----------------------------------------------
This script evaluates the CodonTransformer model with enhanced capabilities:
1. DNAChisel post-processing for sequence polishing
2. Pareto frontier filtering for multi-objective optimization
3. Enhanced beam search with multiple candidates
4. Comprehensive metrics calculation and ablation studies
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
from CodonTransformer.CodonPostProcessing import (
    enhanced_sequence_generation,
    polish_sequence_with_dnachisel,
    apply_pareto_filtering,
    calculate_sequence_metrics,
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


def evaluate_with_enhancements(
    protein_sequence: str,
    model,
    tokenizer,
    device,
    cai_weights: Dict[str, float],
    tai_weights: Dict[str, float],
    codon_frequencies: Dict,
    reference_profile: List[float],
    args,
) -> Dict[str, Any]:
    """
    Evaluate a protein sequence with enhanced generation techniques.
    
    Args:
        protein_sequence: Input protein sequence
        model: Fine-tuned model
        tokenizer: Model tokenizer
        device: PyTorch device
        cai_weights: CAI weights dictionary
        tai_weights: tAI weights dictionary
        codon_frequencies: Codon frequencies dictionary
        reference_profile: Reference profile for DTW calculation
        args: Command line arguments
        
    Returns:
        Dict containing evaluation results for all methods
    """
    results = {}
    
    # 1. Original fine-tuned model (baseline)
    try:
        original_output = predict_dna_sequence(
            protein=protein_sequence,
            organism="Escherichia coli general",
            device=device,
            model=model,
            deterministic=True,
            match_protein=True,
            use_constrained_search=args.use_constrained_search,
            gc_bounds=tuple(args.gc_bounds),
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            diversity_penalty=args.diversity_penalty,
        )
        
        if isinstance(original_output, list):
            original_dna = original_output[0].predicted_dna
        else:
            original_dna = original_output.predicted_dna
            
        results['fine_tuned_original'] = {
            'dna_sequence': original_dna,
            'method': 'fine_tuned_original',
            'enhancement': 'none',
        }
        
    except Exception as e:
        print(f"Warning: Original fine-tuned generation failed: {str(e)}")
        results['fine_tuned_original'] = {
            'dna_sequence': '',
            'method': 'fine_tuned_original',
            'enhancement': 'none',
            'error': str(e),
        }
    
    # 2. Enhanced sequence generation (DNAChisel + Pareto filtering)
    if args.use_enhanced_generation:
        try:
            enhanced_dna, generation_report = enhanced_sequence_generation(
                protein_sequence=protein_sequence,
                model=model,
                tokenizer=tokenizer,
                device=device,
                beam_size=args.enhanced_beam_size,
                gc_bounds=(args.gc_bounds[0] * 100, args.gc_bounds[1] * 100),
                use_dnachisel_polish=args.use_dnachisel,
                use_pareto_filtering=args.use_pareto_filtering,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=reference_profile,
            )
            
            results['fine_tuned_enhanced'] = {
                'dna_sequence': enhanced_dna,
                'method': 'fine_tuned_enhanced',
                'enhancement': 'dnachisel+pareto',
                'generation_report': generation_report,
            }
            
        except Exception as e:
            print(f"Warning: Enhanced generation failed: {str(e)}")
            results['fine_tuned_enhanced'] = {
                'dna_sequence': '',
                'method': 'fine_tuned_enhanced',
                'enhancement': 'dnachisel+pareto',
                'error': str(e),
            }
    
    # 3. DNAChisel post-processing only (ablation study)
    if args.use_dnachisel and 'fine_tuned_original' in results and results['fine_tuned_original']['dna_sequence']:
        try:
            dnachisel_dna, polish_report = polish_sequence_with_dnachisel(
                dna_sequence=results['fine_tuned_original']['dna_sequence'],
                protein_sequence=protein_sequence,
                gc_bounds=(args.gc_bounds[0] * 100, args.gc_bounds[1] * 100),
                maximize_cai=True,
                seed=42,
            )
            
            results['fine_tuned_dnachisel'] = {
                'dna_sequence': dnachisel_dna,
                'method': 'fine_tuned_dnachisel',
                'enhancement': 'dnachisel_only',
                'polish_report': polish_report,
            }
            
        except Exception as e:
            print(f"Warning: DNAChisel post-processing failed: {str(e)}")
            results['fine_tuned_dnachisel'] = {
                'dna_sequence': '',
                'method': 'fine_tuned_dnachisel',
                'enhancement': 'dnachisel_only',
                'error': str(e),
            }
    
    return results


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
    
    return calculate_sequence_metrics(
        dna_sequence=dna_sequence,
        protein_sequence=protein_sequence,
        cai_weights=cai_weights,
        tai_weights=tai_weights,
        codon_frequencies=codon_frequencies,
        reference_profile=reference_profile,
    )


def run_ablation_study(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ablation study to compare different enhancement methods.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        DataFrame with ablation study results
    """
    # Group by protein and calculate improvements
    ablation_results = []
    
    for protein in results_df['protein_sequence'].unique():
        protein_results = results_df[results_df['protein_sequence'] == protein]
        
        # Get baseline (original fine-tuned)
        baseline = protein_results[protein_results['method'] == 'fine_tuned_original']
        if baseline.empty:
            continue
            
        baseline_metrics = baseline.iloc[0]
        
        # Compare each enhancement method
        for method in protein_results['method'].unique():
            if method == 'fine_tuned_original':
                continue
                
            method_results = protein_results[protein_results['method'] == method]
            if method_results.empty:
                continue
                
            method_metrics = method_results.iloc[0]
            
            # Calculate improvements
            improvements = {
                'protein': protein,
                'method': method,
                'enhancement': method_metrics['enhancement'],
                'cai_improvement': method_metrics['cai'] - baseline_metrics['cai'],
                'tai_improvement': method_metrics['tai'] - baseline_metrics['tai'],
                'gc_improvement': abs(method_metrics['gc_content'] - 52) - abs(baseline_metrics['gc_content'] - 52),
                'restriction_sites_improvement': baseline_metrics['restriction_sites'] - method_metrics['restriction_sites'],
                'neg_cis_improvement': baseline_metrics['neg_cis_elements'] - method_metrics['neg_cis_elements'],
                'homopolymer_improvement': baseline_metrics['homopolymer_runs'] - method_metrics['homopolymer_runs'],
                'dtw_improvement': baseline_metrics['dtw_distance'] - method_metrics['dtw_distance'],
                'composite_score_improvement': (
                    (method_metrics['cai'] - baseline_metrics['cai']) * 0.3 +
                    (method_metrics['tai'] - baseline_metrics['tai']) * 0.3 +
                    (abs(baseline_metrics['gc_content'] - 52) - abs(method_metrics['gc_content'] - 52)) * 0.2 +
                    (baseline_metrics['restriction_sites'] - method_metrics['restriction_sites']) * 0.1 +
                    (baseline_metrics['neg_cis_elements'] - method_metrics['neg_cis_elements']) * 0.1
                ),
            }
            
            ablation_results.append(improvements)
    
    return pd.DataFrame(ablation_results)


def main(args):
    """Main function to run the enhanced evaluation."""
    print("=== Enhanced CodonTransformer Evaluation ===")
    
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
    finetuned_model = load_model(model_path=args.checkpoint_path, device=device)
    print(f"Fine-tuned model loaded from {args.checkpoint_path}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    
    # Load base model if comparison requested
    base_model = None
    if args.compare_with_base:
        base_model = load_model(device=device)
        print("Base model loaded from Hugging Face")
    
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
    evaluation_reports = []
    
    print("Starting enhanced evaluation...")
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
        
        # Evaluate with enhancements
        protein_results = evaluate_with_enhancements(
            protein_sequence=protein_sequence,
            model=finetuned_model,
            tokenizer=tokenizer,
            device=device,
            cai_weights=cai_weights,
            tai_weights=tai_weights,
            codon_frequencies=codon_frequencies,
            reference_profile=avg_reference_profile,
            args=args,
        )
        
        # Add base model comparison if requested
        if base_model:
            try:
                base_output = predict_dna_sequence(
                    protein=protein_sequence,
                    organism="Escherichia coli general",
                    device=device,
                    model=base_model,
                    deterministic=True,
                    match_protein=True,
                )
                base_dna = base_output.predicted_dna if not isinstance(base_output, list) else base_output[0].predicted_dna
                
                protein_results['base_model'] = {
                    'dna_sequence': base_dna,
                    'method': 'base_model',
                    'enhancement': 'none',
                }
            except Exception as e:
                print(f"Warning: Base model generation failed: {str(e)}")
        
        # Add naive baseline
        try:
            naive_dna = get_high_frequency_choice_sequence_optimized(
                protein=protein_sequence, codon_frequencies=codon_frequencies
            )
            protein_results['naive_hfc'] = {
                'dna_sequence': naive_dna,
                'method': 'naive_hfc',
                'enhancement': 'none',
            }
        except Exception as e:
            print(f"Warning: Naive HFC generation failed: {str(e)}")
        
        # Calculate metrics for each method
        for method_name, method_result in protein_results.items():
            if 'error' in method_result:
                continue
                
            dna_seq = method_result['dna_sequence']
            if not dna_seq:
                continue
            
            metrics = calculate_comprehensive_metrics(
                dna_sequence=dna_seq,
                protein_sequence=protein_sequence,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=avg_reference_profile,
                ref_sequences=ref_sequences,
            )
            
            # Combine results
            result_row = {
                'protein_id': i,
                'protein_sequence': protein_sequence,
                'protein_length': len(protein_sequence),
                'method': method_name,
                'enhancement': method_result['enhancement'],
                'dna_sequence': dna_seq,
                'dna_length': len(dna_seq),
                **metrics,
            }
            
            # Add generation reports if available
            if 'generation_report' in method_result:
                result_row['generation_report'] = str(method_result['generation_report'])
            if 'polish_report' in method_result:
                result_row['polish_report'] = str(method_result['polish_report'])
            
            all_results.append(result_row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_df.to_csv(args.output_path, index=False)
    print(f"Detailed results saved to {args.output_path}")
    
    # Run ablation study
    if args.run_ablation_study:
        ablation_df = run_ablation_study(results_df)
        ablation_path = args.output_path.replace('.csv', '_ablation.csv')
        ablation_df.to_csv(ablation_path, index=False)
        print(f"Ablation study results saved to {ablation_path}")
        
        # Print summary statistics
        print("\n=== ABLATION STUDY SUMMARY ===")
        for method in ablation_df['method'].unique():
            method_results = ablation_df[ablation_df['method'] == method]
            print(f"\n{method.upper()}:")
            print(f"  CAI improvement: {method_results['cai_improvement'].mean():.4f} ± {method_results['cai_improvement'].std():.4f}")
            print(f"  tAI improvement: {method_results['tai_improvement'].mean():.4f} ± {method_results['tai_improvement'].std():.4f}")
            print(f"  GC improvement: {method_results['gc_improvement'].mean():.4f} ± {method_results['gc_improvement'].std():.4f}")
            print(f"  Restriction sites improvement: {method_results['restriction_sites_improvement'].mean():.2f} ± {method_results['restriction_sites_improvement'].std():.2f}")
            print(f"  Composite score improvement: {method_results['composite_score_improvement'].mean():.4f} ± {method_results['composite_score_improvement'].std():.4f}")
    
    # Print final summary
    print("\n=== EVALUATION COMPLETE ===")
    print(f"Total proteins evaluated: {len(results_df['protein_id'].unique())}")
    print(f"Total sequences generated: {len(results_df)}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced CodonTransformer Evaluation")
    
    # Input/Output paths
    parser.add_argument("--checkpoint_path", type=str, default="models/ecoli-codon-optimizer/finetune_best.ckpt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--test_data_path", type=str, default="data/test_set.json",
                        help="Path to test dataset")
    parser.add_argument("--natural_sequences_path", type=str, default="data/ecoli_processed_genes.csv",
                        help="Path to natural E. coli sequences for CAI calculation")
    parser.add_argument("--output_path", type=str, default="results/enhanced_evaluation_results.csv",
                        help="Path to save evaluation results")
    
    # Model parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--compare_with_base", action="store_true", help="Compare with base model")
    
    # Generation parameters
    parser.add_argument("--use_constrained_search", action="store_true", 
                        help="Use constrained beam search")
    parser.add_argument("--gc_bounds", type=float, nargs=2, default=[0.50, 0.54],
                        help="GC content bounds (min max)")
    parser.add_argument("--beam_size", type=int, default=10,
                        help="Beam size for standard generation")
    parser.add_argument("--length_penalty", type=float, default=1.2,
                        help="Length penalty for beam search")
    parser.add_argument("--diversity_penalty", type=float, default=0.1,
                        help="Diversity penalty for beam search")
    
    # Enhancement parameters
    parser.add_argument("--use_enhanced_generation", action="store_true",
                        help="Use enhanced generation with DNAChisel and Pareto filtering")
    parser.add_argument("--enhanced_beam_size", type=int, default=20,
                        help="Beam size for enhanced generation")
    parser.add_argument("--use_dnachisel", action="store_true",
                        help="Use DNAChisel post-processing")
    parser.add_argument("--use_pareto_filtering", action="store_true",
                        help="Use Pareto frontier filtering")
    
    # Evaluation parameters
    parser.add_argument("--max_test_proteins", type=int, default=0,
                        help="Maximum number of proteins to test (0 for all)")
    parser.add_argument("--run_ablation_study", action="store_true",
                        help="Run ablation study comparing methods")
    
    args = parser.parse_args()
    main(args)