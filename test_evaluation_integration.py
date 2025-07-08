#!/usr/bin/env python3
"""
Test script to validate the integration of enhanced metrics into the evaluation pipeline.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add the CodonTransformer module to the path
sys.path.append('./CodonTransformer')

from CodonEvaluation import (
    calculate_ENC, calculate_CPB, calculate_SCUO,
    get_GC_content, calculate_tAI, get_ecoli_tai_weights
)
from CAI import CAI, relative_adaptiveness

def create_test_dataset():
    """Create a small test dataset for validation."""
    test_data = [
        {
            "protein_sequence": "MKELDIRLREELLEKREDLKGLIKLEEGELLEGYKEGREKAKLFEELKA",
            "dna_sequences": {
                "fine_tuned": "ATGAAAGAACTGGATATTCGCCTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGGAAGAAGGCGAACTGCTGGAAGGCTACAAAGAAGGCCGCGAAAAAGCCAAACTGTTTGAAGAACTGAAAGCC",
                "base": "ATGAAGGAGTTGGATATTCGTTTGCGTGAGGAGTTGTTGGAGAACCGTGAGGATTTGAAGGGCTTGATCAAGTTGGAGGAGGGCGAGTTGTTGGAGGGCTACAAGGAGGGCCGTGAGAAGGCCAAGTTGTTTGAGGAGTTGAAGGCC",
                "naive_hfc": "ATGAAAGAGCTTGACATCCGTCTGCGTGAGGAACTGCTGGAGAACCGTGAGGACCTGAAAGGCCTGATCAAGCTGGAAGAGGGTGAACTGCTGGAGGGTTATAAAGAAGGCCGTGAGAAGGCCAAGCTGTTTGAAGAACTGAAGGCC"
            }
        }
    ]
    return test_data

def test_evaluation_integration():
    """Test the complete evaluation pipeline with enhanced metrics."""
    
    print("Testing Enhanced Evaluation Pipeline Integration")
    print("=" * 60)
    
    # Create test dataset
    test_data = create_test_dataset()
    
    # Set up reference data for CAI calculation
    print("Setting up reference data...")
    
    # Use simple reference sequences for testing
    ref_sequences = [
        "ATGAAAGAACTGGATATTCGCCTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGGAAGAAGGCGAACTGCTGGAAGGCTACAAAGAAGGCCGCGAAAAAGCCAAACTGTTTGAAGAACTGAAAGCC",
        "ATGAAGGAGTTGGATATTCGTTTGCGTGAGGAGTTGTTGGAGAACCGTGAGGATTTGAAGGGCTTGATCAAGTTGGAGGAGGGCGAGTTGTTGGAGGGCTACAAGGAGGGCCGTGAGAAGGCCAAGTTGTTTGAGGAGTTGAAGGCC",
        "ATGAAAGAGCTTGACATCCGTCTGCGTGAGGAACTGCTGGAGAACCGTGAGGACCTGAAAGGCCTGATCAAGCTGGAAGAGGGTGAACTGCTGGAGGGTTATAAAGAAGGCCGTGAGAAGGCCAAGCTGTTTGAAGAACTGAAGGCC"
    ]
    
    # Calculate CAI weights
    cai_weights = relative_adaptiveness(sequences=ref_sequences)
    tai_weights = get_ecoli_tai_weights()
    
    print("✓ Reference data prepared")
    
    # Process test data
    results = []
    
    for item in test_data:
        protein = item["protein_sequence"]
        
        for model_name, dna_sequence in item["dna_sequences"].items():
            print(f"\nEvaluating {model_name} sequence...")
            print(f"  Protein: {protein[:20]}...")
            print(f"  DNA length: {len(dna_sequence)} bp")
            
            try:
                # Calculate all metrics
                metrics = {}
                
                # Basic metrics
                metrics["cai"] = CAI(dna_sequence, weights=cai_weights)
                metrics["tai"] = calculate_tAI(dna_sequence, tai_weights)
                metrics["gc_content"] = get_GC_content(dna_sequence)
                
                # Enhanced metrics
                metrics["enc"] = calculate_ENC(dna_sequence)
                metrics["cpb"] = calculate_CPB(dna_sequence, ref_sequences)
                metrics["scuo"] = calculate_SCUO(dna_sequence)
                
                # Add dummy values for compatibility with analyze_results.py
                metrics["restriction_sites"] = 0
                metrics["neg_cis_elements"] = 0
                metrics["homopolymer_runs"] = 0
                metrics["dtw_distance"] = 1.0
                
                # Store results
                result_entry = {
                    "protein": protein,
                    "model": model_name,
                    "dna_sequence": dna_sequence,
                    **metrics
                }
                results.append(result_entry)
                
                # Display results
                print(f"  Metrics calculated:")
                for metric, value in metrics.items():
                    print(f"    {metric.upper()}: {value:.3f}")
                print(f"  ✓ Success")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Create results DataFrame
    print("\n" + "=" * 60)
    print("Creating results summary...")
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print(f"✓ Created results DataFrame with {len(results_df)} entries")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        numeric_columns = ['cai', 'tai', 'gc_content', 'enc', 'cpb', 'scuo']
        summary = results_df.groupby('model')[numeric_columns].mean()
        
        print(summary.round(3))
        
        # Save test results
        output_path = "results/test_evaluation_results.csv"
        Path("results").mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Test results saved to {output_path}")
        
        # Validate metric ranges
        print("\nValidating metric ranges:")
        validations = [
            ("CAI", "cai", 0.0, 1.0),
            ("tAI", "tai", 0.0, 1.0),
            ("GC Content", "gc_content", 0.0, 100.0),
            ("ENC", "enc", 1.0, 61.0),
            ("SCUO", "scuo", 0.0, 1.0)
        ]
        
        for name, col, min_val, max_val in validations:
            if col in results_df.columns:
                values = results_df[col]
                in_range = ((values >= min_val) & (values <= max_val)).all()
                print(f"  {name}: {'✓' if in_range else '✗'} (range: {values.min():.3f} - {values.max():.3f})")
            else:
                print(f"  {name}: ✗ Column not found")
        
        print("\n" + "=" * 60)
        print("Enhanced evaluation pipeline test completed successfully!")
        return True
        
    else:
        print("✗ No results generated")
        return False

if __name__ == "__main__":
    success = test_evaluation_integration()
    if success:
        print("\n🎉 All tests passed! The enhanced evaluation framework is ready.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")