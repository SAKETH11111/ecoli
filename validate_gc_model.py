#!/usr/bin/env python3
"""
Validation script for GC-controlled CodonTransformer model
Tests robustness on test_set.json and diverse synthetic sequences
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import json
import sys
import os

# Add the CodonTransformer directory to path
sys.path.append('/home/saketh/ecoli')

from CodonTransformer.CodonPrediction import predict_dna_sequence, load_model, load_tokenizer

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence"""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0

def calculate_sliding_window_gc(sequence: str, window_size: int = 50) -> List[float]:
    """Calculate GC content in sliding windows"""
    gc_values = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        gc_values.append(calculate_gc_content(window))
    return gc_values

def dna_to_protein(dna_sequence: str) -> str:
    """Convert DNA sequence to protein sequence"""
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
        if i + 2 < len(dna_sequence):
            codon = dna_sequence[i:i+3]
            protein += codon_table.get(codon, 'X')
    
    return protein.rstrip('*')  # Remove stop codons

def load_test_set() -> List[Dict]:
    """Load test sequences from test_set.json"""
    with open('/home/saketh/ecoli/data/test_set.json', 'r') as f:
        return json.load(f)

def get_synthetic_test_proteins() -> List[Tuple[str, str]]:
    """Return diverse synthetic test proteins"""
    return [
        ("Short_Basic", "MKLLVV"),
        ("Short_Acidic", "DDEEDD"),
        ("Short_Hydrophobic", "VVLLII"),
        ("Short_Polar", "SSTTQQ"),
        ("Medium_Mixed", "MKTVRQERLKSIVRILERSKEPVSGAQ"),
        ("Medium_Repeats", "MKTVRQERLKSIVRILERSKEPVSGAQMKTVRQERLKSIVRILERSKEPVSGAQ"),
        ("Long_Realistic", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGVLDIALMQTGQVQLSLVPRGSQVQHVHLATQV"),
    ]

def validate_model_on_test_set(model_path: str, device: torch.device) -> Dict:
    """Test model on the official test set"""
    print(f"Loading model from: {model_path}")
    
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer()
    
    test_data = load_test_set()
    results = []
    
    print(f"Testing on {len(test_data)} test set sequences...")
    
    for i, test_seq in enumerate(test_data[:20]):  # Test first 20 for speed
        # Extract protein from original DNA
        original_dna = test_seq['codons']
        original_protein = dna_to_protein(original_dna)
        
        if len(original_protein) == 0:
            continue
            
        print(f"\nTest {i+1}/20: {original_protein[:50]}{'...' if len(original_protein) > 50 else ''}")
        
        try:
            # Predict DNA sequence
            prediction = predict_dna_sequence(
                protein=original_protein,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=model,
                deterministic=True
            )
            
            predicted_dna = prediction.predicted_dna
            predicted_protein = dna_to_protein(predicted_dna)
            
            # Calculate metrics
            original_gc = calculate_gc_content(original_dna)
            predicted_gc = calculate_gc_content(predicted_dna)
            
            # Check protein match
            protein_match = original_protein == predicted_protein
            
            # Calculate sliding window GC if sequence is long enough
            if len(predicted_dna) >= 50:
                window_gc = calculate_sliding_window_gc(predicted_dna, 50)
                gc_std = np.std(window_gc)
                gc_min = min(window_gc)
                gc_max = max(window_gc)
            else:
                gc_std = 0.0
                gc_min = predicted_gc
                gc_max = predicted_gc
            
            results.append({
                'test_id': i,
                'original_protein': original_protein,
                'predicted_protein': predicted_protein,
                'protein_match': protein_match,
                'protein_length': len(original_protein),
                'original_dna': original_dna,
                'predicted_dna': predicted_dna,
                'original_gc': original_gc,
                'predicted_gc': predicted_gc,
                'gc_improvement': predicted_gc - original_gc,
                'gc_std': gc_std,
                'gc_min': gc_min,
                'gc_max': gc_max,
                'success': True
            })
            
            print(f"  Original GC: {original_gc:.3f}")
            print(f"  Predicted GC: {predicted_gc:.3f}")
            print(f"  Protein match: {protein_match}")
            print(f"  GC range: [{gc_min:.3f}, {gc_max:.3f}]")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'test_id': i,
                'original_protein': original_protein,
                'predicted_protein': '',
                'protein_match': False,
                'protein_length': len(original_protein),
                'original_dna': original_dna,
                'predicted_dna': '',
                'original_gc': calculate_gc_content(original_dna),
                'predicted_gc': 0.0,
                'gc_improvement': 0.0,
                'gc_std': 0.0,
                'gc_min': 0.0,
                'gc_max': 0.0,
                'success': False
            })
    
    return results

def validate_model_on_synthetic(model_path: str, device: torch.device) -> Dict:
    """Test model on synthetic sequences"""
    print(f"\nTesting on synthetic sequences...")
    
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer()
    
    synthetic_proteins = get_synthetic_test_proteins()
    results = []
    
    for protein_name, protein_seq in synthetic_proteins:
        print(f"\nTesting {protein_name}: {protein_seq}")
        
        try:
            prediction = predict_dna_sequence(
                protein=protein_seq,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=model,
                deterministic=True
            )
            
            predicted_dna = prediction.predicted_dna
            predicted_protein = dna_to_protein(predicted_dna)
            gc_content = calculate_gc_content(predicted_dna)
            
            # Calculate sliding window GC if sequence is long enough
            if len(predicted_dna) >= 50:
                window_gc = calculate_sliding_window_gc(predicted_dna, 50)
                gc_std = np.std(window_gc)
                gc_min = min(window_gc)
                gc_max = max(window_gc)
            else:
                gc_std = 0.0
                gc_min = gc_content
                gc_max = gc_content
            
            protein_match = protein_seq == predicted_protein
            
            results.append({
                'protein_name': protein_name,
                'original_protein': protein_seq,
                'predicted_protein': predicted_protein,
                'protein_match': protein_match,
                'protein_length': len(protein_seq),
                'predicted_dna': predicted_dna,
                'predicted_gc': gc_content,
                'gc_std': gc_std,
                'gc_min': gc_min,
                'gc_max': gc_max,
                'success': True
            })
            
            print(f"  DNA: {predicted_dna}")
            print(f"  GC: {gc_content:.3f}")
            print(f"  Protein match: {protein_match}")
            print(f"  GC range: [{gc_min:.3f}, {gc_max:.3f}]")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'protein_name': protein_name,
                'original_protein': protein_seq,
                'predicted_protein': '',
                'protein_match': False,
                'protein_length': len(protein_seq),
                'predicted_dna': '',
                'predicted_gc': 0.0,
                'gc_std': 0.0,
                'gc_min': 0.0,
                'gc_max': 0.0,
                'success': False
            })
    
    return results

def analyze_results(test_results: List[Dict], synthetic_results: List[Dict]) -> None:
    """Analyze and print validation results"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE VALIDATION RESULTS")
    print(f"{'='*80}")
    
    # Test set analysis
    successful_tests = [r for r in test_results if r['success']]
    print(f"\nTEST SET ANALYSIS:")
    print(f"  Total tests: {len(test_results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Success rate: {len(successful_tests)/len(test_results)*100:.1f}%")
    
    if successful_tests:
        gc_values = [r['predicted_gc'] for r in successful_tests]
        protein_matches = sum(1 for r in successful_tests if r['protein_match'])
        
        print(f"\n  GC CONTENT ANALYSIS:")
        print(f"    Mean predicted GC: {np.mean(gc_values):.3f}")
        print(f"    Std predicted GC: {np.std(gc_values):.3f}")
        print(f"    Min predicted GC: {min(gc_values):.3f}")
        print(f"    Max predicted GC: {max(gc_values):.3f}")
        
        in_range = sum(1 for gc in gc_values if 0.500 <= gc <= 0.540)
        consistency = in_range / len(gc_values) * 100
        print(f"    Target range [0.500-0.540]: {in_range}/{len(gc_values)} ({consistency:.1f}%)")
        
        print(f"\n  PROTEIN FIDELITY:")
        print(f"    Protein matches: {protein_matches}/{len(successful_tests)} ({protein_matches/len(successful_tests)*100:.1f}%)")
        
        # GC improvement analysis
        original_gcs = [r['original_gc'] for r in successful_tests]
        predicted_gcs = [r['predicted_gc'] for r in successful_tests]
        improvements = [r['gc_improvement'] for r in successful_tests]
        
        print(f"\n  GC IMPROVEMENT ANALYSIS:")
        print(f"    Original mean GC: {np.mean(original_gcs):.3f}")
        print(f"    Predicted mean GC: {np.mean(predicted_gcs):.3f}")
        print(f"    Mean improvement: {np.mean(improvements):.3f}")
        print(f"    Std improvement: {np.std(improvements):.3f}")
    
    # Synthetic test analysis
    successful_synthetic = [r for r in synthetic_results if r['success']]
    print(f"\nSYNTHETIC TEST ANALYSIS:")
    print(f"  Total tests: {len(synthetic_results)}")
    print(f"  Successful: {len(successful_synthetic)}")
    print(f"  Success rate: {len(successful_synthetic)/len(synthetic_results)*100:.1f}%")
    
    if successful_synthetic:
        gc_values_syn = [r['predicted_gc'] for r in successful_synthetic]
        protein_matches_syn = sum(1 for r in successful_synthetic if r['protein_match'])
        
        print(f"\n  GC CONTENT ANALYSIS:")
        print(f"    Mean predicted GC: {np.mean(gc_values_syn):.3f}")
        print(f"    Std predicted GC: {np.std(gc_values_syn):.3f}")
        print(f"    Min predicted GC: {min(gc_values_syn):.3f}")
        print(f"    Max predicted GC: {max(gc_values_syn):.3f}")
        
        in_range_syn = sum(1 for gc in gc_values_syn if 0.500 <= gc <= 0.540)
        consistency_syn = in_range_syn / len(gc_values_syn) * 100
        print(f"    Target range [0.500-0.540]: {in_range_syn}/{len(gc_values_syn)} ({consistency_syn:.1f}%)")
        
        print(f"\n  PROTEIN FIDELITY:")
        print(f"    Protein matches: {protein_matches_syn}/{len(successful_synthetic)} ({protein_matches_syn/len(successful_synthetic)*100:.1f}%)")
        
        print(f"\n  DETAILED BREAKDOWN:")
        for r in successful_synthetic:
            status = "✓" if 0.500 <= r['predicted_gc'] <= 0.540 else "✗"
            match_status = "✓" if r['protein_match'] else "✗"
            print(f"    {status} {r['protein_name']}: GC={r['predicted_gc']:.3f}, Match={match_status}")
    
    # Overall summary
    total_tests = len(test_results) + len(synthetic_results)
    total_successful = len(successful_tests) + len(successful_synthetic)
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY:")
    print(f"  Total tests: {total_tests}")
    print(f"  Total successful: {total_successful}")
    print(f"  Overall success rate: {total_successful/total_tests*100:.1f}%")
    
    if successful_tests and successful_synthetic:
        all_gc_values = gc_values + gc_values_syn
        all_in_range = sum(1 for gc in all_gc_values if 0.500 <= gc <= 0.540)
        overall_consistency = all_in_range / len(all_gc_values) * 100
        print(f"  Overall GC consistency: {overall_consistency:.1f}%")
        print(f"  Mean GC across all tests: {np.mean(all_gc_values):.3f}")

def main():
    """Main validation function"""
    print("GC-Controlled CodonTransformer Comprehensive Validation")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model path
    model_path = "/home/saketh/ecoli/checkpoints/lightning_logs/version_7/checkpoints/epoch=14-step=9195.ckpt"
    
    # Run validations
    test_results = validate_model_on_test_set(model_path, device)
    synthetic_results = validate_model_on_synthetic(model_path, device)
    
    # Analyze results
    analyze_results(test_results, synthetic_results)
    
    # Save results
    all_results = {
        'test_set_results': test_results,
        'synthetic_results': synthetic_results,
        'model_path': model_path
    }
    
    # Save to CSV
    test_df = pd.DataFrame(test_results)
    synthetic_df = pd.DataFrame(synthetic_results)
    
    test_df.to_csv('/home/saketh/ecoli/test_set_validation.csv', index=False)
    synthetic_df.to_csv('/home/saketh/ecoli/synthetic_validation.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  Test set: /home/saketh/ecoli/test_set_validation.csv")
    print(f"  Synthetic: /home/saketh/ecoli/synthetic_validation.csv")

if __name__ == "__main__":
    main()