#!/usr/bin/env python3
"""
Benchmark script for CAI/tAI improvements with GC-controlled model
Compares against base model and measures biological relevance
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
from CodonTransformer.CodonEvaluation import get_CSI_weights, get_CSI_value, calculate_tAI, get_ecoli_tai_weights, get_GC_content

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence (as fraction)"""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0

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
    
    return protein.rstrip('*')

def load_test_sequences(limit: int = 10) -> List[Dict]:
    """Load test sequences from test_set.json"""
    with open('/home/saketh/ecoli/data/test_set.json', 'r') as f:
        data = json.load(f)
    return data[:limit]

def benchmark_models(gc_model_path: str, device: torch.device) -> Dict:
    """Benchmark GC-controlled vs base model on CAI/tAI metrics"""
    
    print(f"Loading models...")
    print(f"  GC model: {gc_model_path}")
    print(f"  Base model: Loading from HuggingFace")
    
    # Load models
    gc_model = load_model(gc_model_path, device=device)
    base_model = load_model(device=device)  # No path = loads from HuggingFace
    tokenizer = load_tokenizer()
    
    # Load E. coli CAI weights and tAI weights
    print(f"Loading E. coli reference weights...")
    test_data = load_test_sequences(50)  # Use 50 sequences to compute CAI weights
    
    # Extract DNA sequences for CAI weight calculation
    reference_sequences = []
    for seq_data in test_data:
        dna = seq_data['codons']
        if len(dna) >= 150:  # Only use sequences with at least 50 codons
            reference_sequences.append(dna)
    
    print(f"Computing CAI weights from {len(reference_sequences)} reference sequences...")
    cai_weights = get_CSI_weights(reference_sequences)
    tai_weights = get_ecoli_tai_weights()
    
    # Test sequences
    test_sequences = load_test_sequences(20)  # Test on first 20
    results = []
    
    print(f"\nBenchmarking on {len(test_sequences)} test sequences...")
    
    for i, test_seq in enumerate(test_sequences):
        original_dna = test_seq['codons']
        original_protein = dna_to_protein(original_dna)
        
        if len(original_protein) == 0:
            continue
            
        print(f"\nTest {i+1}/{len(test_sequences)}: {original_protein[:30]}{'...' if len(original_protein) > 30 else ''}")
        
        try:
            # Predict with GC-controlled model
            gc_prediction = predict_dna_sequence(
                protein=original_protein,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=gc_model,
                deterministic=True
            )
            
            # Predict with base model
            base_prediction = predict_dna_sequence(
                protein=original_protein,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=base_model,
                deterministic=True
            )
            
            gc_dna = gc_prediction.predicted_dna
            base_dna = base_prediction.predicted_dna
            
            # Calculate metrics for all sequences
            original_cai = get_CSI_value(original_dna, cai_weights)
            original_tai = calculate_tAI(original_dna, tai_weights)
            original_gc = calculate_gc_content(original_dna)
            
            gc_cai = get_CSI_value(gc_dna, cai_weights)
            gc_tai = calculate_tAI(gc_dna, tai_weights)
            gc_gc = calculate_gc_content(gc_dna)
            
            base_cai = get_CSI_value(base_dna, cai_weights)
            base_tai = calculate_tAI(base_dna, tai_weights)
            base_gc = calculate_gc_content(base_dna)
            
            # Calculate improvements
            gc_cai_improvement = ((gc_cai - original_cai) / original_cai * 100) if original_cai > 0 else 0
            gc_tai_improvement = ((gc_tai - original_tai) / original_tai * 100) if original_tai > 0 else 0
            
            base_cai_improvement = ((base_cai - original_cai) / original_cai * 100) if original_cai > 0 else 0
            base_tai_improvement = ((base_tai - original_tai) / original_tai * 100) if original_tai > 0 else 0
            
            # Store results
            result = {
                'test_id': i,
                'protein': original_protein,
                'protein_length': len(original_protein),
                
                # Original sequence metrics
                'original_cai': original_cai,
                'original_tai': original_tai,
                'original_gc': original_gc,
                
                # GC-controlled model metrics
                'gc_cai': gc_cai,
                'gc_tai': gc_tai,
                'gc_gc': gc_gc,
                'gc_cai_improvement': gc_cai_improvement,
                'gc_tai_improvement': gc_tai_improvement,
                
                # Base model metrics
                'base_cai': base_cai,
                'base_tai': base_tai,
                'base_gc': base_gc,
                'base_cai_improvement': base_cai_improvement,
                'base_tai_improvement': base_tai_improvement,
                
                # Comparisons
                'gc_vs_base_cai': gc_cai - base_cai,
                'gc_vs_base_tai': gc_tai - base_tai,
                'gc_vs_base_gc': gc_gc - base_gc,
                
                'success': True
            }
            
            results.append(result)
            
            print(f"  Original: CAI={original_cai:.3f}, tAI={original_tai:.3f}, GC={original_gc:.3f}")
            print(f"  GC Model: CAI={gc_cai:.3f} (+{gc_cai_improvement:+.1f}%), tAI={gc_tai:.3f} (+{gc_tai_improvement:+.1f}%), GC={gc_gc:.3f}")
            print(f"  Base Model: CAI={base_cai:.3f} (+{base_cai_improvement:+.1f}%), tAI={base_tai:.3f} (+{base_tai_improvement:+.1f}%), GC={base_gc:.3f}")
            print(f"  GC vs Base: ΔCAI={gc_cai-base_cai:+.3f}, ΔtAI={gc_tai-base_tai:+.3f}, ΔGC={gc_gc-base_gc:+.3f}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    return {
        'results': results,
        'cai_weights': cai_weights,
        'tai_weights': tai_weights,
        'reference_sequences_count': len(reference_sequences)
    }

def analyze_benchmark_results(benchmark_data: Dict) -> None:
    """Analyze and print benchmark results"""
    results = benchmark_data['results']
    
    if not results:
        print("No successful benchmark results to analyze!")
        return
    
    print(f"\n{'='*80}")
    print("CAI/tAI BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Successfully benchmarked: {len(results)} sequences")
    print(f"Reference sequences for CAI: {benchmark_data['reference_sequences_count']}")
    
    # Extract metrics
    gc_cai_values = [r['gc_cai'] for r in results]
    gc_tai_values = [r['gc_tai'] for r in results]
    gc_gc_values = [r['gc_gc'] for r in results]
    
    base_cai_values = [r['base_cai'] for r in results]
    base_tai_values = [r['base_tai'] for r in results]
    base_gc_values = [r['base_gc'] for r in results]
    
    original_cai_values = [r['original_cai'] for r in results]
    original_tai_values = [r['original_tai'] for r in results]
    original_gc_values = [r['original_gc'] for r in results]
    
    gc_cai_improvements = [r['gc_cai_improvement'] for r in results]
    gc_tai_improvements = [r['gc_tai_improvement'] for r in results]
    
    base_cai_improvements = [r['base_cai_improvement'] for r in results]
    base_tai_improvements = [r['base_tai_improvement'] for r in results]
    
    # Summary statistics
    print(f"\nAVERAGE METRICS:")
    print(f"  Original:   CAI={np.mean(original_cai_values):.3f}, tAI={np.mean(original_tai_values):.3f}, GC={np.mean(original_gc_values):.3f}")
    print(f"  GC Model:   CAI={np.mean(gc_cai_values):.3f}, tAI={np.mean(gc_tai_values):.3f}, GC={np.mean(gc_gc_values):.3f}")
    print(f"  Base Model: CAI={np.mean(base_cai_values):.3f}, tAI={np.mean(base_tai_values):.3f}, GC={np.mean(base_gc_values):.3f}")
    
    print(f"\nIMPROVEMENTS OVER ORIGINAL:")
    print(f"  GC Model:   CAI={np.mean(gc_cai_improvements):+.1f}%, tAI={np.mean(gc_tai_improvements):+.1f}%")
    print(f"  Base Model: CAI={np.mean(base_cai_improvements):+.1f}%, tAI={np.mean(base_tai_improvements):+.1f}%")
    
    # GC Model vs Base Model comparison
    gc_vs_base_cai = [r['gc_vs_base_cai'] for r in results]
    gc_vs_base_tai = [r['gc_vs_base_tai'] for r in results]
    gc_vs_base_gc = [r['gc_vs_base_gc'] for r in results]
    
    print(f"\nGC MODEL vs BASE MODEL:")
    print(f"  Average ΔCAI: {np.mean(gc_vs_base_cai):+.3f} (std: {np.std(gc_vs_base_cai):.3f})")
    print(f"  Average ΔtAI: {np.mean(gc_vs_base_tai):+.3f} (std: {np.std(gc_vs_base_tai):.3f})")
    print(f"  Average ΔGC:  {np.mean(gc_vs_base_gc):+.3f} (std: {np.std(gc_vs_base_gc):.3f})")
    
    # Count wins
    cai_wins = sum(1 for diff in gc_vs_base_cai if diff > 0)
    tai_wins = sum(1 for diff in gc_vs_base_tai if diff > 0)
    
    print(f"\nWIN RATES (GC Model vs Base Model):")
    print(f"  CAI wins: {cai_wins}/{len(results)} ({cai_wins/len(results)*100:.1f}%)")
    print(f"  tAI wins: {tai_wins}/{len(results)} ({tai_wins/len(results)*100:.1f}%)")
    
    # GC content analysis
    target_range_gc = sum(1 for gc in gc_gc_values if 0.500 <= gc <= 0.540)
    target_range_base = sum(1 for gc in base_gc_values if 0.500 <= gc <= 0.540)
    
    print(f"\nGC CONTENT ANALYSIS:")
    print(f"  GC Model in target range [0.500-0.540]: {target_range_gc}/{len(results)} ({target_range_gc/len(results)*100:.1f}%)")
    print(f"  Base Model in target range [0.500-0.540]: {target_range_base}/{len(results)} ({target_range_base/len(results)*100:.1f}%)")
    
    # Publication-quality metrics
    excellent_cai = sum(1 for cai in gc_cai_values if cai > 0.8)
    excellent_tai = sum(1 for tai in gc_tai_values if tai > 0.5)
    
    print(f"\nPUBLICATION-QUALITY METRICS:")
    print(f"  GC Model with excellent CAI (>0.8): {excellent_cai}/{len(results)} ({excellent_cai/len(results)*100:.1f}%)")
    print(f"  GC Model with excellent tAI (>0.5): {excellent_tai}/{len(results)} ({excellent_tai/len(results)*100:.1f}%)")
    
    # Detailed breakdown for top performing sequences
    print(f"\nTOP 5 PERFORMING SEQUENCES (by CAI improvement):")
    sorted_results = sorted(results, key=lambda x: x['gc_cai_improvement'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. CAI: {r['original_cai']:.3f} → {r['gc_cai']:.3f} (+{r['gc_cai_improvement']:.1f}%), "
              f"tAI: {r['original_tai']:.3f} → {r['gc_tai']:.3f} (+{r['gc_tai_improvement']:.1f}%), "
              f"GC: {r['original_gc']:.3f} → {r['gc_gc']:.3f}")

def main():
    """Main benchmark function"""
    print("CAI/tAI Benchmark: GC-Controlled vs Base Model")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model paths
    gc_model_path = "/home/saketh/ecoli/checkpoints/lightning_logs/version_7/checkpoints/epoch=14-step=9195.ckpt"
    
    # Run benchmark
    benchmark_data = benchmark_models(gc_model_path, device)
    
    # Analyze results
    analyze_benchmark_results(benchmark_data)
    
    # Save results
    results_df = pd.DataFrame(benchmark_data['results'])
    results_df.to_csv('/home/saketh/ecoli/cai_tai_benchmark.csv', index=False)
    print(f"\nResults saved to: /home/saketh/ecoli/cai_tai_benchmark.csv")

if __name__ == "__main__":
    main()