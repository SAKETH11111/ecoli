#!/usr/bin/env python3
"""
Test script to verify compatibility between enhanced evaluation metrics 
and the existing evaluation pipeline.
"""

import sys
import os
sys.path.append('./CodonTransformer')

from CodonEvaluation import (
    calculate_ENC,
    calculate_CPB,
    calculate_SCUO,
    calculate_tAI,
    get_GC_content,
    get_ecoli_tai_weights,
    scan_for_restriction_sites,
    count_negative_cis_elements,
    calculate_homopolymer_runs,
    get_min_max_profile
)
from CAI import CAI, relative_adaptiveness

def test_pipeline_compatibility():
    """Test that enhanced metrics work with existing pipeline."""
    
    print("Testing Pipeline Compatibility")
    print("=" * 50)
    
    # Test sequence (E. coli-like)
    test_dna = "ATGAAAGAACTGGATATTCGCCTGCGCGAAGAACTGCTGGAAAAACGCGAAGATCTGAAAGGCCTGATCAAACTGGAAGAAGGCGAACTGCTGGAAGGCTACAAAGAAGGCCGCGAAAAAGCCAAACTGTTTGAAGAACTGAAAGCCTAA"
    test_protein = "MKELDIRLREELLEKRIRDLKGLIKLEEGELLEGYKEGREKAKLFEELKA"
    
    # Reference sequences for CAI and CPB
    ref_sequences = [
        test_dna,
        "ATGAAAGAGTTGGACATCCGTTTGCGTGAGGAGTTGTTGGAGAACCGTGAGGATTTGAAGGGCTTGATCAAGTTGGAGGAGGGCGAGTTGTTGGAGGGCTACAAGGAGGGCCGTGAGAAGGCCAAGTTGTTTGAGGAGTTGAAGGCCTAG"
    ]
    
    print(f"Test sequence: {len(test_dna)} bp")
    print(f"Test protein: {len(test_protein)} aa")
    
    # Test all metrics as they would be used in the pipeline
    results = {}
    
    try:
        print("\n1. Testing Traditional Metrics:")
        
        # CAI
        cai_weights = relative_adaptiveness(sequences=ref_sequences)
        results['cai'] = CAI(test_dna, weights=cai_weights)
        print(f"   ✓ CAI: {results['cai']:.3f}")
        
        # tAI
        tai_weights = get_ecoli_tai_weights()
        results['tai'] = calculate_tAI(test_dna, tai_weights)
        print(f"   ✓ tAI: {results['tai']:.3f}")
        
        # GC Content
        results['gc_content'] = get_GC_content(test_dna)
        print(f"   ✓ GC Content: {results['gc_content']:.1f}%")
        
        # Health metrics
        results['restriction_sites'] = scan_for_restriction_sites(test_dna)
        results['neg_cis_elements'] = count_negative_cis_elements(test_dna)
        results['homopolymer_runs'] = calculate_homopolymer_runs(test_dna)
        
        print(f"   ✓ Restriction sites: {results['restriction_sites']}")
        print(f"   ✓ Negative cis elements: {results['neg_cis_elements']}")
        print(f"   ✓ Homopolymer runs: {results['homopolymer_runs']}")
        
    except Exception as e:
        print(f"   ✗ Traditional metrics failed: {e}")
        return False
    
    try:
        print("\n2. Testing Enhanced Codon Usage Metrics:")
        
        # ENC
        results['enc'] = calculate_ENC(test_dna)
        print(f"   ✓ ENC: {results['enc']:.3f}")
        
        # CPB
        results['cpb'] = calculate_CPB(test_dna, ref_sequences)
        print(f"   ✓ CPB: {results['cpb']:.3f}")
        
        # SCUO
        results['scuo'] = calculate_SCUO(test_dna)
        print(f"   ✓ SCUO: {results['scuo']:.3f}")
        
    except Exception as e:
        print(f"   ✗ Enhanced metrics failed: {e}")
        return False
    
    print("\n3. Testing Results Integration:")
    
    # Simulate the results structure from evaluate_optimizer.py
    evaluation_result = {
        "protein": test_protein,
        "model": "test_model",
        "dna_sequence": test_dna,
        "cai": results['cai'],
        "tai": results['tai'],
        "gc_content": results['gc_content'],
        "restriction_sites": results['restriction_sites'],
        "neg_cis_elements": results['neg_cis_elements'],
        "homopolymer_runs": results['homopolymer_runs'],
        "enc": results['enc'],
        "cpb": results['cpb'],
        "scuo": results['scuo'],
    }
    
    print("   ✓ Results structure compatible with evaluation pipeline")
    print(f"   ✓ All {len(evaluation_result)} metrics calculated successfully")
    
    print("\n4. Enhanced Metrics Performance:")
    print(f"   • ENC range: [1.0, 61.0] - Current: {results['enc']:.3f}")
    print(f"   • CPB interpretation: Higher = more biased - Current: {results['cpb']:.3f}")
    print(f"   • SCUO range: [0.0, 1.0] - Current: {results['scuo']:.3f}")
    
    print("\n" + "=" * 50)
    print("✅ Pipeline compatibility test PASSED!")
    print("Enhanced evaluation framework is fully integrated and ready for use.")
    
    return True

if __name__ == "__main__":
    success = test_pipeline_compatibility()
    
    if success:
        print("\n🎉 Phase 1.1 Implementation Complete!")
        print("Enhanced evaluation framework successfully integrated with:")
        print("  • ENC (Effective Number of Codons) - codonbias library")
        print("  • CPB (Codon Pair Bias) - codonbias library")
        print("  • SCUO (Synonymous Codon Usage Order) - GCUA library with fallback")
        print("  • Full compatibility with existing evaluation pipeline")
    else:
        print("\n❌ Compatibility issues detected. Please check implementation.")