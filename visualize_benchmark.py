#!/usr/bin/env python3
"""
Visualization script for CAI/tAI benchmark results
Creates publication-quality plots similar to analyze_results.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import wilcoxon
from typing import List, Dict

def create_results_dir():
    """Create results directory if it doesn't exist"""
    if not os.path.exists('benchmark_plots'):
        os.makedirs('benchmark_plots')

def load_benchmark_data() -> pd.DataFrame:
    """Load and prepare benchmark data for visualization"""
    # Load the benchmark results
    df = pd.read_csv('/home/saketh/ecoli/cai_tai_benchmark.csv')
    
    # Create long format for better plotting
    data_rows = []
    
    for _, row in df.iterrows():
        # Original sequence
        data_rows.append({
            'sequence_id': row['test_id'],
            'protein_length': row['protein_length'],
            'model': 'Original',
            'cai': row['original_cai'],
            'tai': row['original_tai'],
            'gc_content': row['original_gc'] * 100,  # Convert to percentage
            'cai_improvement': 0.0,
            'tai_improvement': 0.0
        })
        
        # GC-controlled model
        data_rows.append({
            'sequence_id': row['test_id'],
            'protein_length': row['protein_length'],
            'model': 'GC-Controlled',
            'cai': row['gc_cai'],
            'tai': row['gc_tai'],
            'gc_content': row['gc_gc'] * 100,
            'cai_improvement': row['gc_cai_improvement'],
            'tai_improvement': row['gc_tai_improvement']
        })
        
        # Base model
        data_rows.append({
            'sequence_id': row['test_id'],
            'protein_length': row['protein_length'],
            'model': 'Base',
            'cai': row['base_cai'],
            'tai': row['base_tai'],
            'gc_content': row['base_gc'] * 100,
            'cai_improvement': row['base_cai_improvement'],
            'tai_improvement': row['base_tai_improvement']
        })
    
    return pd.DataFrame(data_rows)

def plot_cai_comparison(df: pd.DataFrame):
    """Plot CAI comparison across models"""
    plt.figure(figsize=(12, 8))
    
    # Box plot with individual points
    sns.boxplot(data=df, x='model', y='cai', palette='Set2')
    sns.stripplot(data=df, x='model', y='cai', color='black', alpha=0.6, size=4)
    
    plt.title('Codon Adaptation Index (CAI) Comparison\nGC-Controlled vs Base vs Original', fontsize=16, fontweight='bold')
    plt.ylabel('CAI Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0.5, 1.0)
    
    # Add horizontal lines for benchmarks
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Excellent CAI (>0.8)')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good CAI (>0.7)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_plots/cai_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tai_comparison(df: pd.DataFrame):
    """Plot tAI comparison across models"""
    plt.figure(figsize=(12, 8))
    
    # Box plot with individual points
    sns.boxplot(data=df, x='model', y='tai', palette='Set2')
    sns.stripplot(data=df, x='model', y='tai', color='black', alpha=0.6, size=4)
    
    plt.title('tRNA Adaptation Index (tAI) Comparison\nGC-Controlled vs Base vs Original', fontsize=16, fontweight='bold')
    plt.ylabel('tAI Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0.2, 0.6)
    
    # Add horizontal lines for benchmarks
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Excellent tAI (>0.5)')
    plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Good tAI (>0.4)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_plots/tai_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_gc_content_analysis(df: pd.DataFrame):
    """Plot GC content distribution analysis"""
    plt.figure(figsize=(12, 8))
    
    # Box plot with individual points
    sns.boxplot(data=df, x='model', y='gc_content', palette='Set2')
    sns.stripplot(data=df, x='model', y='gc_content', color='black', alpha=0.6, size=4)
    
    plt.title('GC Content Distribution\nGC-Controlled vs Base vs Original', fontsize=16, fontweight='bold')
    plt.ylabel('GC Content (%)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(30, 70)
    
    # Add target range
    plt.axhspan(50, 54, alpha=0.2, color='green', label='Target Range (50-54%)')
    plt.axhline(y=52, color='green', linestyle='-', alpha=0.8, label='Optimal GC (52%)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_plots/gc_content_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_metrics(df: pd.DataFrame):
    """Plot improvement metrics for optimized models"""
    # Filter out original sequences for improvement plots
    df_opt = df[df['model'] != 'Original'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CAI improvement
    sns.barplot(data=df_opt, x='model', y='cai_improvement', ax=axes[0], palette='Set2')
    axes[0].set_title('CAI Improvement Over Original', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('CAI Improvement (%)', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, model in enumerate(['Base', 'GC-Controlled']):
        mean_val = df_opt[df_opt['model'] == model]['cai_improvement'].mean()
        axes[0].text(i, mean_val + 1, f'{mean_val:.1f}%', ha='center', fontweight='bold')
    
    # tAI improvement
    sns.barplot(data=df_opt, x='model', y='tai_improvement', ax=axes[1], palette='Set2')
    axes[1].set_title('tAI Improvement Over Original', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('tAI Improvement (%)', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, model in enumerate(['Base', 'GC-Controlled']):
        mean_val = df_opt[df_opt['model'] == model]['tai_improvement'].mean()
        axes[1].text(i, mean_val + 1, f'{mean_val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_plots/improvement_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_analysis(df: pd.DataFrame):
    """Plot correlation between different metrics"""
    # Create pivot for correlation analysis
    pivot_df = df.pivot(index='sequence_id', columns='model', values=['cai', 'tai', 'gc_content'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CAI vs tAI scatter for each model
    for i, model in enumerate(['Original', 'GC-Controlled']):
        row = i // 2
        col = i % 2
        
        model_data = df[df['model'] == model]
        sns.scatterplot(data=model_data, x='cai', y='tai', ax=axes[row, col], alpha=0.7, s=60)
        axes[row, col].set_title(f'CAI vs tAI - {model}', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('CAI Score', fontsize=10)
        axes[row, col].set_ylabel('tAI Score', fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = model_data['cai'].corr(model_data['tai'])
        axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[row, col].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # GC vs CAI
    model_data = df[df['model'] == 'GC-Controlled']
    sns.scatterplot(data=model_data, x='gc_content', y='cai', ax=axes[1, 0], alpha=0.7, s=60, color='orange')
    axes[1, 0].set_title('GC Content vs CAI - GC-Controlled', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('GC Content (%)', fontsize=10)
    axes[1, 0].set_ylabel('CAI Score', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=52, color='green', linestyle='--', alpha=0.7, label='Target GC')
    axes[1, 0].legend()
    
    # GC vs tAI
    sns.scatterplot(data=model_data, x='gc_content', y='tai', ax=axes[1, 1], alpha=0.7, s=60, color='red')
    axes[1, 1].set_title('GC Content vs tAI - GC-Controlled', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('GC Content (%)', fontsize=10)
    axes[1, 1].set_ylabel('tAI Score', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=52, color='green', linestyle='--', alpha=0.7, label='Target GC')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_plots/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(df: pd.DataFrame):
    """Create a comprehensive performance summary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Summary statistics by model
    summary_stats = df.groupby('model').agg({
        'cai': ['mean', 'std'],
        'tai': ['mean', 'std'],
        'gc_content': ['mean', 'std']
    }).round(3)
    
    models = ['Original', 'Base', 'GC-Controlled']
    metrics = ['CAI', 'tAI', 'GC Content (%)']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # Bar plots for each metric
    for i, (metric, col) in enumerate(zip(['cai', 'tai', 'gc_content'], metrics)):
        ax = axes[i//2, i%2] if i < 3 else None
        if ax is None:
            continue
            
        means = [summary_stats.loc[model, (metric, 'mean')] for model in models]
        stds = [summary_stats.loc[model, (metric, 'std')] for model in models]
        
        bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{col} - Mean ± Std', fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01*max(means),
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Summary table in the fourth subplot
    axes[1, 1].axis('off')
    
    # Calculate key statistics
    gc_model_data = df[df['model'] == 'GC-Controlled']
    base_model_data = df[df['model'] == 'Base']
    original_data = df[df['model'] == 'Original']
    
    target_gc_count = len(gc_model_data[(gc_model_data['gc_content'] >= 50) & (gc_model_data['gc_content'] <= 54)])
    excellent_cai_count = len(gc_model_data[gc_model_data['cai'] > 0.8])
    good_tai_count = len(gc_model_data[gc_model_data['tai'] > 0.4])
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    GC-Controlled Model:
    • Target GC Range (50-54%): {target_gc_count}/20 ({target_gc_count/20*100:.0f}%)
    • Excellent CAI (>0.8): {excellent_cai_count}/20 ({excellent_cai_count/20*100:.0f}%)
    • Good tAI (>0.4): {good_tai_count}/20 ({good_tai_count/20*100:.0f}%)
    
    Average Improvements:
    • CAI: +{gc_model_data["cai_improvement"].mean():.1f}%
    • tAI: +{gc_model_data["tai_improvement"].mean():.1f}%
    
    Mean Values:
    • CAI: {gc_model_data["cai"].mean():.3f}
    • tAI: {gc_model_data["tai"].mean():.3f}
    • GC: {gc_model_data["gc_content"].mean():.1f}%
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('benchmark_plots/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_report(df: pd.DataFrame) -> str:
    """Generate statistical significance report"""
    report = "=== STATISTICAL ANALYSIS REPORT ===\n\n"
    
    # Basic statistics
    for model in ['Original', 'Base', 'GC-Controlled']:
        model_data = df[df['model'] == model]
        report += f"{model} Model Statistics:\n"
        report += f"  CAI: {model_data['cai'].mean():.3f} ± {model_data['cai'].std():.3f}\n"
        report += f"  tAI: {model_data['tai'].mean():.3f} ± {model_data['tai'].std():.3f}\n"
        report += f"  GC:  {model_data['gc_content'].mean():.1f}% ± {model_data['gc_content'].std():.1f}%\n\n"
    
    # Pairwise comparisons (if models differ)
    pivot_df = df.pivot(index='sequence_id', columns='model', values=['cai', 'tai', 'gc_content'])
    
    model_pairs = [('GC-Controlled', 'Original'), ('Base', 'Original'), ('GC-Controlled', 'Base')]
    metrics = ['cai', 'tai', 'gc_content']
    
    report += "Wilcoxon Signed-Rank Test Results:\n"
    for metric in metrics:
        report += f"\n{metric.upper()}:\n"
        for model1, model2 in model_pairs:
            try:
                if model1 in pivot_df[metric].columns and model2 in pivot_df[metric].columns:
                    data1 = pivot_df[metric][model1].dropna()
                    data2 = pivot_df[metric][model2].dropna()
                    
                    if len(data1) > 0 and len(data2) > 0 and not data1.equals(data2):
                        stat, p_value = wilcoxon(data1, data2)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        report += f"  {model1} vs {model2}: p={p_value:.4f} {significance}\n"
                    else:
                        report += f"  {model1} vs {model2}: identical or insufficient data\n"
            except Exception as e:
                report += f"  {model1} vs {model2}: test failed ({str(e)})\n"
    
    return report

def main():
    """Main visualization function"""
    print("Creating benchmark visualizations...")
    
    # Create output directory
    create_results_dir()
    
    # Load and prepare data
    df = load_benchmark_data()
    print(f"Loaded data for {len(df)} data points across {df['model'].nunique()} models")
    
    # Generate plots
    print("Generating plots...")
    plot_cai_comparison(df)
    print("  ✓ CAI comparison plot")
    
    plot_tai_comparison(df)
    print("  ✓ tAI comparison plot")
    
    plot_gc_content_analysis(df)
    print("  ✓ GC content analysis plot")
    
    plot_improvement_metrics(df)
    print("  ✓ Improvement metrics plot")
    
    plot_correlation_analysis(df)
    print("  ✓ Correlation analysis plot")
    
    plot_performance_summary(df)
    print("  ✓ Performance summary plot")
    
    # Generate statistical report
    print("Generating statistical report...")
    report = generate_statistical_report(df)
    
    with open('benchmark_plots/statistical_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    
    print(f"\nAll plots saved to: benchmark_plots/")
    print("Files generated:")
    print("  - cai_comparison.png")
    print("  - tai_comparison.png") 
    print("  - gc_content_distribution.png")
    print("  - improvement_metrics.png")
    print("  - correlation_analysis.png")
    print("  - performance_summary.png")
    print("  - statistical_report.txt")

if __name__ == "__main__":
    main()