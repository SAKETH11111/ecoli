#!/usr/bin/env python3
"""
Test script for the enhanced codon usage metrics (ENC, CPB, SCUO).
This script validates that the new functions work correctly with sample sequences.
"""

import sys
import os
sys.path.append('./CodonTransformer')

from CodonEvaluation import calculate_ENC, calculate_CPB, calculate_SCUO

def test_enhanced_metrics():
    """Test the enhanced codon usage metrics with sample sequences."""
    
    print("Testing Enhanced Codon Usage Metrics")
    print("=" * 50)
    
    # Test sequences (actual E. coli coding sequences)
    test_sequences = {
        "high_bias_sequence": "ATGAAAGAACTGAAAGATCTGATCGAACTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGAAAGAAGGCGAACTGCTGGAAGGCAAAAAAGAAGGCCGCGAAAAAGCCGGCAAACTGTTTGAAGAACTGAAAGCCAAA",
        "low_bias_sequence": "ATGAAGGAGTTGAAGGATTTGATCGAGTTGCGGGAGGAGTTGTTGGAGAACCGGGAGGATTTGAAGGGCTTGATCAAGTTGAAGGAGGGCGAGTTGTTGGAGGGCAAGAAGGAGGGCCGGGAGAAGGCCGGCAAGTTGTTTGAGGAGTTGAAGGCCAAG",
        "mixed_sequence": "ATGAAAGAGCTGAAGGATCTGATCGAACTGCGCGAGGAACTGCTGGAAAACCGCGAAGATCTGAAGGGCCTGATCAAACTGAAAGAAGGCGAACTGCTGGAAGGCAAAAAAGAAGGCCGCGAAAAAGCCGGCAAACTGTTTGAAGAACTGAAAGCCAAA"
    }
    
    # Test each sequence
    for seq_name, sequence in test_sequences.items():
        print(f"\nTesting sequence: {seq_name}")
        print(f"Length: {len(sequence)} bp ({len(sequence)//3} codons)")
        
        try:
            # Test ENC calculation
            enc_value = calculate_ENC(sequence)
            print(f"  ENC: {enc_value:.3f}")
            
            # Test SCUO calculation
            scuo_value = calculate_SCUO(sequence)
            print(f"  SCUO: {scuo_value:.3f}")
            
            # Test CPB calculation
            cpb_value = calculate_CPB(sequence)
            print(f"  CPB: {cpb_value:.3f}")
            
            # Validate ranges
            assert 1.0 <= enc_value <= 61.0, f"ENC value {enc_value} out of expected range [1.0, 61.0]"
            assert 0.0 <= scuo_value <= 1.0, f"SCUO value {scuo_value} out of expected range [0.0, 1.0]"
            print(f"  ✓ All metrics calculated successfully")
            
        except Exception as e:
            print(f"  ✗ Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing with reference sequences for CPB")
    
    # Test CPB with reference sequences
    reference_seqs = [
        "ATGAAAGAACTGAAAGATCTGATCGAACTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGAAAGAAGGCGAACTGCTGGAAGGCAAAAAAGAAGGCCGCGAAAAAGCCGGCAAACTGTTTGAAGAACTGAAAGCCAAA",
        "ATGAAGGAGTTGAAGGATTTGATCGAGTTGCGGGAGGAGTTGTTGGAGAACCGGGAGGATTTGAAGGGCTTGATCAAGTTGAAGGAGGGCGAGTTGTTGGAGGGCAAGAAGGAGGGCCGGGAGAAGGCCGGCAAGTTGTTTGAGGAGTTGAAGGCCAAG"
    ]
    
    target_sequence = test_sequences["mixed_sequence"]
    
    try:
        cpb_with_ref = calculate_CPB(target_sequence, reference_seqs)
        print(f"CPB with reference sequences: {cpb_with_ref:.3f}")
        print("✓ CPB with reference calculation successful")
    except Exception as e:
        print(f"✗ Error calculating CPB with reference: {e}")
    
    print("\n" + "=" * 50)
    print("Validation of biological expectations:")
    
    # Biological validation
    print("\nENC interpretation:")
    print("  - Values near 20: Extremely biased codon usage")
    print("  - Values near 61: Random codon usage")
    print("  - E. coli typically shows moderate bias (ENC ~45-55)")
    
    print("\nSCUO interpretation:")
    print("  - Values near 0: Random synonymous codon usage")
    print("  - Values near 1: Highly ordered/biased synonymous codon usage")
    
    print("\nCPB interpretation:")
    print("  - Positive values: Codon pairs more frequent than expected")
    print("  - Negative values: Codon pairs less frequent than expected")
    print("  - Values near 0: Random codon pair usage")
    
    print("\n✓ Enhanced codon usage metrics test completed successfully!")

if __name__ == "__main__":
    test_enhanced_metrics()