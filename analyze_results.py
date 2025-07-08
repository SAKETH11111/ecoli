import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
import numpy as np

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load the results
results_df = pd.read_csv('results/evaluation_results.csv')
results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check if new metrics are available
has_enc = 'enc' in results_df.columns
has_cpb = 'cpb' in results_df.columns
has_scuo = 'scuo' in results_df.columns


# 1. Codon Adaptation Index (CAI) Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='model', y='cai')
plt.title('Codon Adaptation Index (CAI) Comparison')
plt.ylabel('CAI Score')
plt.xlabel('Model')
plt.savefig('results/cai_comparison.png')
plt.close()

# 2. tRNA Adaptation Index (tAI) Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='model', y='tai')
plt.title('tRNA Adaptation Index (tAI) Comparison')
plt.ylabel('tAI Score')
plt.xlabel('Model')
plt.savefig('results/tai_comparison.png')
plt.close()

# 3. GC Content Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='model', y='gc_content')
plt.title('GC Content Distribution')
plt.ylabel('GC Content (%)')
plt.xlabel('Model')
plt.ylim(40, 65)
plt.axhline(y=45, color='r', linestyle='--', label='Healthy Range (45-60%)')
plt.axhline(y=60, color='r', linestyle='--')
plt.legend()
plt.savefig('results/gc_content_distribution.png')
plt.close()

# 4. Sequence Health Metrics
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.barplot(data=results_df, x='model', y='restriction_sites', ax=axes[0])
axes[0].set_title('Restriction Site Occurrences')

sns.barplot(data=results_df, x='model', y='neg_cis_elements', ax=axes[1])
axes[1].set_title('Negative Cis-Regulatory Elements')

sns.barplot(data=results_df, x='model', y='homopolymer_runs', ax=axes[2])
axes[2].set_title('Homopolymer Runs (>8bp)')

plt.tight_layout()
plt.savefig('results/sequence_health_metrics.png')
plt.close()

# 5. DTW Distance Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='model', y='dtw_distance')
plt.title('DTW Distance from Natural %MinMax Profile')
plt.ylabel('DTW Distance')
plt.xlabel('Model')
plt.savefig('results/dtw_distance_comparison.png')
plt.close()

# 6. Delta CAI vs Delta DTW Scatter Plot
pivot_df = results_df.pivot(index='protein', columns='model', values=['cai', 'dtw_distance'])
delta_df = pd.DataFrame()
delta_df['delta_cai'] = pivot_df[('cai', 'fine_tuned')] - pivot_df[('cai', 'base')]
delta_df['delta_dtw'] = pivot_df[('dtw_distance', 'fine_tuned')] - pivot_df[('dtw_distance', 'base')]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=delta_df, x='delta_cai', y='delta_dtw')
plt.title('Trade-off: ΔCAI vs ΔDTW (Fine-tuned - Base)')
plt.xlabel('CAI Improvement (ΔCAI)')
plt.ylabel('DTW Distance Change (ΔDTW)')
plt.grid(True)
plt.savefig('results/delta_cai_vs_delta_dtw.png')
plt.close()


# 7. Statistical Significance Tests
print("--- Statistical Significance Tests (Wilcoxon signed-rank) ---")

model_pairs = [('fine_tuned', 'base'), ('fine_tuned', 'naive_hfc'), ('base', 'naive_hfc')]
metrics = ['cai', 'tai', 'dtw_distance', 'gc_content', 'restriction_sites', 'neg_cis_elements', 'homopolymer_runs']

for metric in metrics:
    print(f"\n--- Metric: {metric.upper()} ---")
    pivot_df = results_df.pivot(index='protein', columns='model', values=metric).dropna()
    for model1, model2 in model_pairs:
        if model1 not in pivot_df.columns or model2 not in pivot_df.columns:
            continue
            
        m1_data = pivot_df[model1]
        m2_data = pivot_df[model2]

        if m1_data.var() < 1e-10 or m2_data.var() < 1e-10:
            print(f"Skipping {model1} vs. {model2} due to zero variance in one of the models.")
            continue
            
        stat, p_value = wilcoxon(m1_data, m2_data)
        print(f"{model1} vs. {model2}: p-value = {p_value:.4f}")

# 7. Additional Plots
# GC Content violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=results_df, x='model', y='gc_content')
plt.title('GC Content Distribution (Violin Plot)')
plt.ylabel('GC Content (%)')
plt.xlabel('Model')
plt.axhline(y=45, color='r', linestyle='--', label='Healthy Range (45-60%)')
plt.axhline(y=60, color='r', linestyle='--')
plt.legend()
plt.savefig('results/gc_content_violin.png')
plt.close()

# tAI violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=results_df, x='model', y='tai')
plt.title('tAI Distribution (Violin Plot)')
plt.ylabel('tAI Score')
plt.xlabel('Model')
plt.savefig('results/tai_violin.png')
plt.close()

# CAI vs GC Content scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='gc_content', y='cai', hue='model', alpha=0.7)
plt.title('CAI vs. GC Content')
plt.xlabel('GC Content (%)')
plt.ylabel('CAI Score')
plt.grid(True)
plt.savefig('results/cai_vs_gc_content.png')
plt.close()


# Enhanced Codon Usage Analysis Plots
if has_enc:
    # ENC Analysis
    plt.figure(figsize=(12, 8))
    
    # ENC comparison barplot
    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x='model', y='enc')
    plt.title('Effective Number of Codons (ENC)')
    plt.ylabel('ENC Value')
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Low bias threshold')
    plt.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Medium bias threshold')
    plt.legend()
    
    # ENC violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(data=results_df, x='model', y='enc')
    plt.title('ENC Distribution (Violin Plot)')
    plt.ylabel('ENC Value')
    
    # ENC vs CAI scatter
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=results_df, x='enc', y='cai', hue='model', alpha=0.7)
    plt.title('ENC vs CAI Relationship')
    plt.xlabel('ENC Value')
    plt.ylabel('CAI Score')
    
    # ENC vs GC content
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=results_df, x='enc', y='gc_content', hue='model', alpha=0.7)
    plt.title('ENC vs GC Content')
    plt.xlabel('ENC Value')
    plt.ylabel('GC Content (%)')
    
    plt.tight_layout()
    plt.savefig('results/enc_analysis.png')
    plt.close()

if has_cpb:
    # CPB Analysis
    plt.figure(figsize=(12, 6))
    
    # CPB comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='model', y='cpb')
    plt.title('Codon Pair Bias (CPB)')
    plt.ylabel('CPB Value')
    
    # CPB distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(data=results_df, x='model', y='cpb')
    plt.title('CPB Distribution')
    plt.ylabel('CPB Value')
    
    plt.tight_layout()
    plt.savefig('results/cpb_analysis.png')
    plt.close()

if has_scuo:
    # SCUO Analysis
    plt.figure(figsize=(12, 6))
    
    # SCUO comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='model', y='scuo')
    plt.title('Synonymous Codon Usage Order (SCUO)')
    plt.ylabel('SCUO Value')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random usage')
    plt.legend()
    
    # SCUO vs CAI relationship
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=results_df, x='scuo', y='cai', hue='model', alpha=0.7)
    plt.title('SCUO vs CAI Relationship')
    plt.xlabel('SCUO Value')
    plt.ylabel('CAI Score')
    
    plt.tight_layout()
    plt.savefig('results/scuo_analysis.png')
    plt.close()

# Comprehensive codon usage metrics heatmap
if has_enc and has_cpb and has_scuo:
    # Create correlation matrix of all codon usage metrics
    codon_metrics = ['cai', 'tai', 'enc', 'cpb', 'scuo', 'gc_content']
    available_metrics = [m for m in codon_metrics if m in results_df.columns]
    
    if len(available_metrics) >= 3:
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix for each model
        for i, model in enumerate(results_df['model'].unique()):
            model_data = results_df[results_df['model'] == model][available_metrics]
            
            plt.subplot(2, 2, i+1)
            correlation_matrix = model_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': .8})
            plt.title(f'{model.replace("_", " ").title()} - Metric Correlations')
        
        plt.tight_layout()
        plt.savefig('results/codon_metrics_correlations.png')
        plt.close()

# Multi-metric comparison radar chart
if has_enc and has_cpb and has_scuo:
    from math import pi
    
    # Normalize metrics for radar chart (0-1 scale)
    metrics_for_radar = ['cai', 'tai', 'enc', 'scuo']
    available_radar_metrics = [m for m in metrics_for_radar if m in results_df.columns]
    
    if len(available_radar_metrics) >= 3:
        fig, axes = plt.subplots(1, len(results_df['model'].unique()), 
                                figsize=(5*len(results_df['model'].unique()), 5),
                                subplot_kw=dict(projection='polar'))
        
        if len(results_df['model'].unique()) == 1:
            axes = [axes]
        
        for idx, model in enumerate(results_df['model'].unique()):
            model_data = results_df[results_df['model'] == model]
            
            # Calculate mean values and normalize
            values = []
            for metric in available_radar_metrics:
                if metric == 'enc':
                    # For ENC, lower is better (more biased), so invert
                    normalized_val = 1 - (model_data[metric].mean() / 61.0)
                else:
                    # For others, normalize to 0-1 scale
                    normalized_val = model_data[metric].mean()
                    if metric in ['cai', 'tai', 'scuo']:
                        # These are already 0-1 scale typically
                        pass
                    else:
                        # Normalize other metrics
                        min_val = results_df[metric].min()
                        max_val = results_df[metric].max()
                        if max_val > min_val:
                            normalized_val = (normalized_val - min_val) / (max_val - min_val)
                        else:
                            normalized_val = 0.5
                
                values.append(max(0, min(1, normalized_val)))  # Clamp to 0-1
            
            # Set up angles for radar chart
            angles = [n / float(len(available_radar_metrics)) * 2 * pi for n in range(len(available_radar_metrics))]
            angles += angles[:1]  # Complete the circle
            values += values[:1]  # Complete the circle
            
            # Plot
            axes[idx].plot(angles, values, 'o-', linewidth=2, label=model)
            axes[idx].fill(angles, values, alpha=0.25)
            axes[idx].set_xticks(angles[:-1])
            axes[idx].set_xticklabels([m.upper() for m in available_radar_metrics])
            axes[idx].set_ylim(0, 1)
            axes[idx].set_title(f'{model.replace("_", " ").title()} Performance Profile')
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/performance_radar_charts.png')
        plt.close()

# 8. Save Summary Data to JSON
p_values_dict = {}
# Update metrics list to include new ones
extended_metrics = metrics.copy()
if has_enc:
    extended_metrics.append('enc')
if has_cpb:
    extended_metrics.append('cpb')
if has_scuo:
    extended_metrics.append('scuo')

for metric in extended_metrics:
    if metric not in results_df.columns:
        continue
    p_values_dict[metric] = {}
    pivot_df = results_df.pivot(index='protein', columns='model', values=metric).dropna()
    for model1, model2 in model_pairs:
        if model1 not in pivot_df.columns or model2 not in pivot_df.columns:
            continue
        m1_data = pivot_df[model1]
        m2_data = pivot_df[model2]
        if m1_data.var() < 1e-10 or m2_data.var() < 1e-10:
            p_values_dict[metric][f"{model1}_vs_{model2}"] = 'skipped (zero variance)'
            continue
        stat, p_value = wilcoxon(m1_data, m2_data)
        p_values_dict[metric][f"{model1}_vs_{model2}"] = p_value

summary_data = {
    "mean_metrics": results_df.groupby('model')[extended_metrics].mean().to_dict(),
    "statistical_tests": p_values_dict,
    "new_metrics_available": {
        "enc": has_enc,
        "cpb": has_cpb,
        "scuo": has_scuo
    }
}
import json
with open("results/evaluation_summary.json", "w") as f:
    json.dump(summary_data, f, indent=4)

print("\nAnalysis complete. Plots and summary JSON saved to the 'results' directory.")
if has_enc or has_cpb or has_scuo:
    print("\nEnhanced codon usage analysis plots generated:")
    if has_enc:
        print("  ✓ ENC (Effective Number of Codons) analysis")
    if has_cpb:
        print("  ✓ CPB (Codon Pair Bias) analysis")
    if has_scuo:
        print("  ✓ SCUO (Synonymous Codon Usage Order) analysis")
    if has_enc and has_cpb and has_scuo:
        print("  ✓ Comprehensive codon metrics correlation heatmaps")
        print("  ✓ Multi-metric performance radar charts")
else:
    print("\nNote: Enhanced codon usage metrics (ENC, CPB, SCUO) not found in results.")
    print("Run the evaluation pipeline with updated CodonEvaluation.py to generate these metrics.")