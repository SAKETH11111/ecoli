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


# 8. Save Summary Data to JSON
p_values_dict = {}
for metric in metrics:
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
    "mean_metrics": results_df.groupby('model')[metrics].mean().to_dict(),
    "statistical_tests": p_values_dict,
}
import json
with open("results/evaluation_summary.json", "w") as f:
    json.dump(summary_data, f, indent=4)

print("\nAnalysis complete. Plots and summary JSON saved to the 'results' directory.")