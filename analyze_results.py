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

# Filter to only the three key methods we want to compare
key_methods = ['base_model', 'fine_tuned_original', 'naive_hfc']
results_df = results_df[results_df['method'].isin(key_methods)]

print("=== ANALYSIS SUMMARY ===")
print(f"Total sequences analyzed: {len(results_df)}")
print(f"Methods: {results_df['method'].unique()}")
print(f"Proteins per method: {results_df.groupby('method').size()}")

# 1. Codon Adaptation Index (CAI) Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='method', y='cai')
plt.title('Codon Adaptation Index (CAI) Comparison')
plt.ylabel('CAI Score')
plt.xlabel('Method')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/cai_comparison.png')
plt.close()

# 2. tRNA Adaptation Index (tAI) Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='method', y='tai')
plt.title('tRNA Adaptation Index (tAI) Comparison')
plt.ylabel('tAI Score')
plt.xlabel('Method')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/tai_comparison.png')
plt.close()

# 3. GC Content Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='method', y='gc_content')
plt.title('GC Content Distribution')
plt.ylabel('GC Content (%)')
plt.xlabel('Method')
plt.ylim(40, 65)
plt.axhline(y=45, color='r', linestyle='--', label='Healthy Range (45-60%)')
plt.axhline(y=60, color='r', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/gc_content_distribution.png')
plt.close()

# 4. Sequence Health Metrics
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.barplot(data=results_df, x='method', y='restriction_sites', ax=axes[0])
axes[0].set_title('Restriction Site Occurrences')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(data=results_df, x='method', y='neg_cis_elements', ax=axes[1])
axes[1].set_title('Negative Cis-Regulatory Elements')
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(data=results_df, x='method', y='homopolymer_runs', ax=axes[2])
axes[2].set_title('Homopolymer Runs (>8bp)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/sequence_health_metrics.png')
plt.close()

# 5. DTW Distance Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='method', y='dtw_distance')
plt.title('DTW Distance from Natural %MinMax Profile')
plt.ylabel('DTW Distance')
plt.xlabel('Method')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/dtw_distance_comparison.png')
plt.close()

# 6. Delta CAI vs Delta DTW Scatter Plot
pivot_df = results_df.pivot(index='protein_id', columns='method', values=['cai', 'dtw_distance'])
delta_df = pd.DataFrame()

# Calculate deltas if both methods exist
if ('cai', 'base_model') in pivot_df.columns and ('cai', 'fine_tuned_original') in pivot_df.columns:
    delta_df['delta_cai'] = pivot_df[('cai', 'fine_tuned_original')] - pivot_df[('cai', 'base_model')]
    delta_df['delta_dtw'] = pivot_df[('dtw_distance', 'fine_tuned_original')] - pivot_df[('dtw_distance', 'base_model')]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=delta_df, x='delta_cai', y='delta_dtw')
    plt.title('Trade-off: ΔCAI vs ΔDTW (Fine-tuned - Base)')
    plt.xlabel('CAI Improvement (ΔCAI)')
    plt.ylabel('DTW Distance Change (ΔDTW)')
    plt.grid(True)
    plt.savefig('results/delta_cai_vs_delta_dtw.png')
    plt.close()

# 7. Statistical Significance Tests
print("\n--- Statistical Significance Tests (Wilcoxon signed-rank) ---")

method_pairs = [('fine_tuned_original', 'base_model'), ('fine_tuned_original', 'naive_hfc'), ('base_model', 'naive_hfc')]
metrics = ['cai', 'tai', 'dtw_distance', 'gc_content', 'restriction_sites', 'neg_cis_elements', 'homopolymer_runs']

for metric in metrics:
    print(f"\n--- Metric: {metric.upper()} ---")
    pivot_df = results_df.pivot(index='protein_id', columns='method', values=metric).dropna()
    for method1, method2 in method_pairs:
        if method1 not in pivot_df.columns or method2 not in pivot_df.columns:
            continue
            
        m1_data = pivot_df[method1]
        m2_data = pivot_df[method2]

        if m1_data.var() < 1e-10 or m2_data.var() < 1e-10:
            print(f"Skipping {method1} vs. {method2} due to zero variance in one of the methods.")
            continue
            
        stat, p_value = wilcoxon(m1_data, m2_data)
        print(f"{method1} vs. {method2}: p-value = {p_value:.4f}")

# 8. Additional Plots
# GC Content violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=results_df, x='method', y='gc_content')
plt.title('GC Content Distribution (Violin Plot)')
plt.ylabel('GC Content (%)')
plt.xlabel('Method')
plt.axhline(y=45, color='r', linestyle='--', label='Healthy Range (45-60%)')
plt.axhline(y=60, color='r', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/gc_content_violin.png')
plt.close()

# tAI violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=results_df, x='method', y='tai')
plt.title('tAI Distribution (Violin Plot)')
plt.ylabel('tAI Score')
plt.xlabel('Method')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/tai_violin.png')
plt.close()

# CAI vs GC Content scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='gc_content', y='cai', hue='method', alpha=0.7)
plt.title('CAI vs. GC Content')
plt.xlabel('GC Content (%)')
plt.ylabel('CAI Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/cai_vs_gc_content.png')
plt.close()

# Enhanced Codon Usage Analysis Plots
if has_enc:
    # ENC Analysis
    plt.figure(figsize=(12, 8))
    
    # ENC comparison barplot
    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x='method', y='enc')
    plt.title('Effective Number of Codons (ENC)')
    plt.ylabel('ENC Value')
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Low bias threshold')
    plt.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Medium bias threshold')
    plt.legend()
    plt.xticks(rotation=45)
    
    # ENC violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(data=results_df, x='method', y='enc')
    plt.title('ENC Distribution (Violin Plot)')
    plt.ylabel('ENC Value')
    plt.xticks(rotation=45)
    
    # ENC vs CAI scatter
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=results_df, x='enc', y='cai', hue='method', alpha=0.7)
    plt.title('ENC vs CAI Relationship')
    plt.xlabel('ENC Value')
    plt.ylabel('CAI Score')
    
    # ENC vs GC content
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=results_df, x='enc', y='gc_content', hue='method', alpha=0.7)
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
    sns.barplot(data=results_df, x='method', y='cpb')
    plt.title('Codon Pair Bias (CPB)')
    plt.ylabel('CPB Value')
    plt.xticks(rotation=45)
    
    # CPB distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(data=results_df, x='method', y='cpb')
    plt.title('CPB Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/cpb_analysis.png')
    plt.close()

if has_scuo:
    # SCUO Analysis
    plt.figure(figsize=(12, 6))
    
    # SCUO comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='method', y='scuo')
    plt.title('Synonymous Codon Usage Order (SCUO)')
    plt.ylabel('SCUO Value')
    plt.xticks(rotation=45)
    
    # SCUO distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(data=results_df, x='method', y='scuo')
    plt.title('SCUO Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/scuo_analysis.png')
    plt.close()

# 9. Performance Radar Chart
def create_radar_chart(data, methods, metrics):
    """Create a radar chart comparing methods across multiple metrics."""
    # Normalize metrics to 0-1 scale for radar chart
    normalized_data = data.copy()
    for metric in metrics:
        if metric in normalized_data.columns:
            min_val = normalized_data[metric].min()
            max_val = normalized_data[metric].max()
            if max_val > min_val:
                normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
    
    # Calculate mean values for each method
    radar_data = normalized_data.groupby('method')[metrics].mean()
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each method
    colors = ['red', 'blue', 'green']
    for i, method in enumerate(methods):
        if method in radar_data.index:
            values = radar_data.loc[method].values.tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Comparison Radar Chart', size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('results/performance_radar_charts.png')
    plt.close()

# Create radar chart for key metrics
key_metrics = ['cai', 'tai', 'gc_content', 'restriction_sites', 'neg_cis_elements', 'homopolymer_runs']
create_radar_chart(results_df, key_methods, key_metrics)

# 10. Summary Statistics
print("\n=== SUMMARY STATISTICS ===")
summary_stats = results_df.groupby('method')[['cai', 'tai', 'gc_content', 'restriction_sites', 'neg_cis_elements', 'homopolymer_runs']].agg(['mean', 'std']).round(4)
print(summary_stats)

print("\n=== ANALYSIS COMPLETE ===")
print("Generated plots saved in results/ directory:")
print("- cai_comparison.png")
print("- tai_comparison.png") 
print("- gc_content_distribution.png")
print("- sequence_health_metrics.png")
print("- dtw_distance_comparison.png")
print("- delta_cai_vs_delta_dtw.png")
print("- gc_content_violin.png")
print("- tai_violin.png")
print("- cai_vs_gc_content.png")
print("- performance_radar_charts.png")
if has_enc:
    print("- enc_analysis.png")
if has_cpb:
    print("- cpb_analysis.png")
if has_scuo:
    print("- scuo_analysis.png")