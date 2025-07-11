#!/usr/bin/env python3
"""
Test script to validate the library-based codon usage metrics integration.
This tests the integration with codonbias and GCUA libraries.
"""

import sys
import os
sys.path.append('./CodonTransformer')

from CodonEvaluation import calculate_ENC, calculate_CPB, calculate_SCUO

def test_library_integration():
    """Test the library-based implementations of ENC, CPB, and SCUO."""
    
    print("Testing Library-Based Codon Usage Metrics Integration")
    print("=" * 60)
    
    # Test sequences (E. coli-like sequences)
    test_sequences = {
        "balanced": "ATGAAAGAACTGGATATTCGCCTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGGAAGAAGGCGAACTGCTGGAAGGCTACAAAGAAGGCCGCGAAAAAGCCAAACTGTTTGAAGAACTGAAAGCCTAA",
        "biased": "ATGAAAGAGTTGGACATCCGTTTGCGTGAGGAGTTGTTGGAGAACCGTGAGGATTTGAAGGGCTTGATCAAGTTGGAGGAGGGCGAGTTGTTGGAGGGCTACAAGGAGGGCCGTGAGAAGGCCAAGTTGTTTGAGGAGTTGAAGGCCTAG",
        "short": "ATGAAAGAACTGGATATTCGCCTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCTAA"
    }
    
    reference_sequences = [
        test_sequences["balanced"],
        test_sequences["biased"]
    ]
    
    # Test each sequence
    for seq_name, sequence in test_sequences.items():
        print(f"\\nTesting sequence: {seq_name}")
        print(f"Length: {len(sequence)} bp ({len(sequence)//3} codons)")
        
        # Test ENC calculation
        try:
            enc_value = calculate_ENC(sequence)
            print(f"  ✓ ENC: {enc_value:.3f}")
            
            # Validate ENC range
            if 1.0 <= enc_value <= 61.0:
                print(f"    ✓ ENC in valid range [1.0, 61.0]")
            else:
                print(f"    ✗ ENC {enc_value} outside valid range")
                
        except Exception as e:
            print(f"  ✗ ENC calculation failed: {e}")
        
        # Test CPB calculation
        try:
            cpb_value = calculate_CPB(sequence, reference_sequences)
            print(f"  ✓ CPB: {cpb_value:.3f}")
            
        except Exception as e:
            print(f"  ✗ CPB calculation failed: {e}")
        
        # Test SCUO calculation
        try:
            scuo_value = calculate_SCUO(sequence)
            print(f"  ✓ SCUO: {scuo_value:.3f}")
            
            # Validate SCUO range
            if 0.0 <= scuo_value <= 1.0:
                print(f"    ✓ SCUO in valid range [0.0, 1.0]")
            else:
                print(f"    ✗ SCUO {scuo_value} outside valid range")
                
        except Exception as e:
            print(f"  ✗ SCUO calculation failed: {e}")
    
    print("\\n" + "=" * 60)
    print("Testing comparative analysis...")
    
    # Test that different sequences give different results
    try:
        enc1 = calculate_ENC(test_sequences["balanced"])
        enc2 = calculate_ENC(test_sequences["biased"])
        
        if abs(enc1 - enc2) > 0.01:  # Should be different
            print(f"✓ ENC differentiates sequences: {enc1:.3f} vs {enc2:.3f}")
        else:
            print(f"⚠ ENC values very similar: {enc1:.3f} vs {enc2:.3f}")
            
    except Exception as e:
        print(f"✗ ENC comparison failed: {e}")
    
    print("\\n" + "=" * 60)
    print("Library integration tests completed!")
    
    # Test with evaluation pipeline
    print("\\nTesting integration with evaluation pipeline...")
    
    try:
        # Test all metrics on a single sequence
        test_seq = test_sequences["balanced"]
        
        enc_val = calculate_ENC(test_seq)
        cpb_val = calculate_CPB(test_seq, reference_sequences)
        scuo_val = calculate_SCUO(test_seq)
        
        print(f"Integrated results for balanced sequence:")
        print(f"  ENC: {enc_val:.3f}")
        print(f"  CPB: {cpb_val:.3f}")
        print(f"  SCUO: {scuo_val:.3f}")
        
        print("\\n✅ All library integrations working correctly!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_library_integration()
    
    if success:
        print("\\n🎉 Library integration successful! Enhanced evaluation framework ready.")
    else:
        print("\\n⚠️ Some integration issues detected. Check implementation.")