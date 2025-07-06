#!/usr/bin/env python3
"""
Complete analysis script for GC-controlled CodonTransformer
Combines validation, benchmarking, DTW calculation, and visualization
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import sys
import os
from scipy.stats import wilcoxon

sys.path.append('/home/saketh/ecoli')

from CodonTransformer.CodonPrediction import predict_dna_sequence, load_model, load_tokenizer
from CodonTransformer.CodonEvaluation import (
    get_CSI_value, calculate_tAI, get_min_max_profile, calculate_dtw_distance,
    scan_for_restriction_sites, count_negative_cis_elements, 
    calculate_homopolymer_runs, get_CSI_weights, get_ecoli_tai_weights
)
from CodonTransformer.CodonData import get_codon_frequencies

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

def main():
    print("="*80)
    print("COMPLETE ANALYSIS: GC-CONTROLLED CODONTRANSFORMER")
    print("Base Model vs GC-Controlled Model - Full Pipeline")
    print("="*80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # Load models
    print("\nLoading models...")
    gc_model_path = "/home/saketh/ecoli/checkpoints/lightning_logs/version_7/checkpoints/epoch=14-step=9195.ckpt"
    gc_model = load_model(gc_model_path, device=device)
    base_model = load_model(device=device)  # HuggingFace base
    tokenizer = load_tokenizer()
    
    # Load test proteins
    print("Loading test proteins...")
    with open('/home/saketh/ecoli/data/test_set.json', 'r') as f:
        test_data = json.load(f)
    
    test_proteins = []
    for item in test_data[:30]:  # Use 30 for comprehensive analysis
        dna = item['codons']
        protein = dna_to_protein(dna)
        if len(protein) >= 10 and len(protein) <= 300:
            test_proteins.append((protein, dna))
    
    print(f"Testing on {len(test_proteins)} proteins")
    
    # Setup codon frequencies and reference profile for DTW
    print("Setting up codon frequencies and reference profile...")
    reference_sequences = [item['codons'] for item in test_data[:100] if len(item['codons']) >= 150]
    codon_frequencies = get_codon_frequencies(reference_sequences, organism="Escherichia coli general")
    
    # Calculate reference profile for DTW
    reference_profiles = []
    for seq in reference_sequences[:30]:
        try:
            profile = get_min_max_profile(seq, codon_frequencies)
            if profile and not all(v is None for v in profile):
                clean_profile = [v for v in profile if v is not None and not np.isnan(v)]
                if clean_profile:
                    reference_profiles.append(clean_profile)
        except:
            continue
    
    if reference_profiles:
        max_len = max(len(p) for p in reference_profiles)
        padded_profiles = [
            np.pad(np.array(p), (0, max_len - len(p)), "constant", constant_values=np.nan)
            for p in reference_profiles
        ]
        avg_reference_profile = np.nanmean(padded_profiles, axis=0)
        print(f"Created reference profile from {len(reference_profiles)} sequences")
    else:
        print("ERROR: No valid reference profiles found")
        return
    
    # Generate sequences and calculate all metrics
    results = []
    
    for i, (protein, original_dna) in enumerate(test_proteins):
        print(f"\nProcessing protein {i+1}/{len(test_proteins)}: {protein[:30]}...")
        
        try:
            # Base model prediction
            base_pred = predict_dna_sequence(
                protein=protein,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=base_model,
                deterministic=True
            )
            base_dna = base_pred.predicted_dna
            
            # GC-controlled model prediction
            gc_pred = predict_dna_sequence(
                protein=protein,
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=gc_model,
                deterministic=True
            )
            gc_dna = gc_pred.predicted_dna
            
            # Calculate all metrics for both sequences
            for model_name, dna_seq in [("base", base_dna), ("fine_tuned", gc_dna)]:
                print(f"  Calculating metrics for {model_name}...")
                
                # Core metrics
                # Get reference sequences for CAI calculation
                ref_sequences = [item['codons'] for item in test_data[:100] if len(item['codons']) >= 150]
                cai_weights = get_CSI_weights(ref_sequences)
                cai_score = get_CSI_value(dna_seq, cai_weights)
                # For tAI, we need to get weights first
                tai_weights = get_ecoli_tai_weights()
                tai_score = calculate_tAI(dna_seq, tai_weights)
                gc_content = (dna_seq.count('G') + dna_seq.count('C')) / len(dna_seq) * 100
                
                # DTW distance
                try:
                    seq_profile = get_min_max_profile(dna_seq, codon_frequencies)
                    if seq_profile and not all(v is None for v in seq_profile):
                        clean_seq_profile = [v for v in seq_profile if v is not None and not np.isnan(v)]
                        if clean_seq_profile:
                            dtw_dist = calculate_dtw_distance(clean_seq_profile, avg_reference_profile)
                            dtw_distance = dtw_dist if not np.isinf(dtw_dist) else np.nan
                        else:
                            dtw_distance = np.nan
                    else:
                        dtw_distance = np.nan
                except Exception as e:
                    dtw_distance = np.nan
                
                # Health metrics
                try:
                    restriction_sites = scan_for_restriction_sites(dna_seq)
                    neg_cis_elements = count_negative_cis_elements(dna_seq)
                    homopolymer_runs = calculate_homopolymer_runs(dna_seq)
                except Exception as e:
                    restriction_sites = 0
                    neg_cis_elements = 0
                    homopolymer_runs = 0
                
                result = {
                    'protein': protein[:50],
                    'model': model_name,
                    'dna_sequence': dna_seq,
                    'cai': cai_score,
                    'tai': tai_score,
                    'gc_content': gc_content,
                    'dtw_distance': dtw_distance,
                    'restriction_sites': restriction_sites,
                    'neg_cis_elements': neg_cis_elements,
                    'homopolymer_runs': homopolymer_runs
                }
                results.append(result)
                
                print(f"    CAI: {cai_score:.3f}, tAI: {tai_score:.3f}, GC: {gc_content:.1f}%")
                print(f"    DTW: {dtw_distance:.0f}, Restriction: {restriction_sites}, Homopoly: {homopolymer_runs}")
                
        except Exception as e:
            print(f"  ERROR processing protein {i+1}: {e}")
            continue
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('/home/saketh/ecoli/analysis_results/complete_analysis.csv', index=False)
    print(f"\nResults saved to analysis_results/complete_analysis.csv")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # 1. CAI Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='model', y='cai', hue='model', legend=False)
    plt.title('Codon Adaptation Index (CAI) Comparison\nBase vs GC-Controlled Model')
    plt.ylabel('CAI Score')
    plt.xlabel('Model')
    plt.xticks([0, 1], ['Base (HuggingFace)', 'GC-Controlled'])
    
    # Add value labels
    for i, model in enumerate(['base', 'fine_tuned']):
        model_data = results_df[results_df['model'] == model]['cai']
        if not model_data.empty:
            mean_val = model_data.mean()
            plt.text(i, mean_val + mean_val*0.02, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/cai_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. tAI Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='model', y='tai', hue='model', legend=False)
    plt.title('tRNA Adaptation Index (tAI) Comparison\nBase vs GC-Controlled Model')
    plt.ylabel('tAI Score')
    plt.xlabel('Model')
    plt.xticks([0, 1], ['Base (HuggingFace)', 'GC-Controlled'])
    
    # Add value labels
    for i, model in enumerate(['base', 'fine_tuned']):
        model_data = results_df[results_df['model'] == model]['tai']
        if not model_data.empty:
            mean_val = model_data.mean()
            plt.text(i, mean_val + mean_val*0.02, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/tai_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. GC Content Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='model', y='gc_content', hue='model', legend=False)
    plt.title('GC Content Distribution\nBase vs GC-Controlled Model')
    plt.ylabel('GC Content (%)')
    plt.xlabel('Model')
    plt.ylim(40, 65)
    plt.axhline(y=50, color='g', linestyle='--', label='Target Min (50%)')
    plt.axhline(y=54, color='g', linestyle='--', label='Target Max (54%)')
    plt.axhline(y=52, color='r', linestyle='-', label='Optimal (52%)', linewidth=2)
    plt.xticks([0, 1], ['Base (HuggingFace)', 'GC-Controlled'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/gc_content_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. DTW Distance Comparison
    if not results_df['dtw_distance'].isna().all():
        plt.figure(figsize=(10, 6))
        plot_df = results_df.dropna(subset=['dtw_distance'])
        
        if not plot_df.empty:
            sns.barplot(data=plot_df, x='model', y='dtw_distance', hue='model', legend=False)
            plt.title('DTW Distance from Natural %MinMax Profile\nBase vs GC-Controlled Model')
            plt.ylabel('DTW Distance')
            plt.xlabel('Model')
            plt.xticks([0, 1], ['Base (HuggingFace)', 'GC-Controlled'])
            
            # Add value labels
            for i, model in enumerate(['base', 'fine_tuned']):
                model_data = plot_df[plot_df['model'] == model]['dtw_distance']
                if not model_data.empty:
                    mean_val = model_data.mean()
                    plt.text(i, mean_val + mean_val*0.05, f'{mean_val:.0f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('/home/saketh/ecoli/analysis_results/dtw_distance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. CAI vs GC Content scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='gc_content', y='cai', hue='model', alpha=0.7, s=60)
    plt.title('CAI vs. GC Content\nBase vs GC-Controlled Model')
    plt.xlabel('GC Content (%)')
    plt.ylabel('CAI Score')
    plt.axvline(x=52, color='red', linestyle='--', alpha=0.7, label='Target GC (52%)')
    plt.axvspan(50, 54, alpha=0.1, color='green', label='Target Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/cai_vs_gc_content.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. tAI vs GC Content scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='gc_content', y='tai', hue='model', alpha=0.7, s=60)
    plt.title('tAI vs. GC Content\nBase vs GC-Controlled Model')
    plt.xlabel('GC Content (%)')
    plt.ylabel('tAI Score')
    plt.axvline(x=52, color='red', linestyle='--', alpha=0.7, label='Target GC (52%)')
    plt.axvspan(50, 54, alpha=0.1, color='green', label='Target Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/tai_vs_gc_content.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Health metrics comparison
    health_metrics = ['restriction_sites', 'neg_cis_elements', 'homopolymer_runs']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(health_metrics):
        if metric in results_df.columns:
            sns.barplot(data=results_df, x='model', y=metric, ax=axes[i], hue='model', legend=False)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xlabel('Model')
            axes[i].set_xticklabels(['Base', 'GC-Controlled'])
    
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/health_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Violin plots for distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # CAI violin
    sns.violinplot(data=results_df, x='model', y='cai', ax=axes[0,0], hue='model', legend=False)
    axes[0,0].set_title('CAI Distribution')
    axes[0,0].set_xticklabels(['Base', 'GC-Controlled'])
    
    # tAI violin
    sns.violinplot(data=results_df, x='model', y='tai', ax=axes[0,1], hue='model', legend=False)
    axes[0,1].set_title('tAI Distribution')
    axes[0,1].set_xticklabels(['Base', 'GC-Controlled'])
    
    # GC Content violin
    sns.violinplot(data=results_df, x='model', y='gc_content', ax=axes[1,0], hue='model', legend=False)
    axes[1,0].set_title('GC Content Distribution')
    axes[1,0].set_xticklabels(['Base', 'GC-Controlled'])
    axes[1,0].axhline(y=52, color='r', linestyle='-', alpha=0.7)
    
    # DTW violin (if available)
    if not results_df['dtw_distance'].isna().all():
        plot_df = results_df.dropna(subset=['dtw_distance'])
        if not plot_df.empty:
            sns.violinplot(data=plot_df, x='model', y='dtw_distance', ax=axes[1,1], hue='model', legend=False)
            axes[1,1].set_title('DTW Distance Distribution')
            axes[1,1].set_xticklabels(['Base', 'GC-Controlled'])
    
    plt.tight_layout()
    plt.savefig('/home/saketh/ecoli/analysis_results/distribution_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical Analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    metrics = ['cai', 'tai', 'gc_content']
    if not results_df['dtw_distance'].isna().all():
        metrics.append('dtw_distance')
    
    for metric in metrics:
        print(f"\n--- {metric.upper()} ---")
        
        # Get paired data
        pivot_df = results_df.pivot(index='protein', columns='model', values=metric).dropna()
        
        if 'base' in pivot_df.columns and 'fine_tuned' in pivot_df.columns:
            base_data = pivot_df['base']
            gc_data = pivot_df['fine_tuned']
            
            # Summary statistics
            print(f"Base model: {base_data.mean():.4f} ± {base_data.std():.4f}")
            print(f"GC-controlled: {gc_data.mean():.4f} ± {gc_data.std():.4f}")
            
            # Improvement calculation
            improvement = (gc_data.mean() - base_data.mean()) / base_data.mean() * 100
            print(f"Improvement: {improvement:+.2f}%")
            
            # Statistical significance
            if len(base_data) > 1 and len(gc_data) > 1:
                try:
                    _, p_value = wilcoxon(base_data, gc_data)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"Statistical significance: p={p_value:.4f} {significance}")
                except:
                    print("Statistical test failed")
    
    # Win rate analysis
    print(f"\n--- WIN RATE ANALYSIS ---")
    for metric in ['cai', 'tai']:
        pivot_metric = results_df.pivot(index='protein', columns='model', values=metric).dropna()
        if 'base' in pivot_metric.columns and 'fine_tuned' in pivot_metric.columns:
            wins = (pivot_metric['fine_tuned'] > pivot_metric['base']).sum()
            total = len(pivot_metric)
            win_rate = wins / total * 100
            print(f"{metric.upper()} win rate: {wins}/{total} ({win_rate:.1f}%)")
    
    # GC target compliance
    gc_pivot = results_df.pivot(index='protein', columns='model', values='gc_content').dropna()
    if 'base' in gc_pivot.columns and 'fine_tuned' in gc_pivot.columns:
        base_in_target = ((gc_pivot['base'] >= 50) & (gc_pivot['base'] <= 54)).sum()
        gc_in_target = ((gc_pivot['fine_tuned'] >= 50) & (gc_pivot['fine_tuned'] <= 54)).sum()
        total = len(gc_pivot)
        
        print(f"\n--- GC TARGET COMPLIANCE (50-54%) ---")
        print(f"Base model: {base_in_target}/{total} ({base_in_target/total*100:.1f}%)")
        print(f"GC-Controlled: {gc_in_target}/{total} ({gc_in_target/total*100:.1f}%)")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: analysis_results/")
    print(f"Generated plots:")
    print(f"  - cai_comparison.png")
    print(f"  - tai_comparison.png")
    print(f"  - gc_content_distribution.png")
    print(f"  - dtw_distance_comparison.png")
    print(f"  - cai_vs_gc_content.png")
    print(f"  - tai_vs_gc_content.png")
    print(f"  - health_metrics_comparison.png")
    print(f"  - distribution_violin_plots.png")
    print(f"  - complete_analysis.csv")

if __name__ == "__main__":
    main()