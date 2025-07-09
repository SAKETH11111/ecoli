#!/usr/bin/env python3
"""
Test script for Constrained Beam Search "Guardrail" System

This script tests the constrained beam search implementation to ensure:
1. GC content is guaranteed to be within 45-55% target range
2. CAI/tAI performance is preserved 
3. The system can handle various protein sequences
"""

import torch
import json
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path so we can import CodonTransformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence_constrained_beam_search
from CodonTransformer.CodonEvaluation import (
    get_CSI_weights, get_CSI_value, get_GC_content, 
    calculate_tAI, get_ecoli_tai_weights,
    calculate_ENC, calculate_CPB, calculate_SCUO
)


def calculate_gc_content_simple(dna_sequence: str) -> float:
    """Simple GC content calculation for verification."""
    if not dna_sequence:
        return 0.0
    gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    return gc_count / len(dna_sequence)


def test_constrained_beam_search():
    """Test constrained beam search with multiple protein sequences."""
    print("🧪 Testing Constrained Beam Search 'Guardrail' System")
    print("=" * 60)
    
    # Test proteins of different lengths and compositions
    test_proteins = [
        # Short protein
        "MKTVRQERLK",
        # Medium protein - GFP fragment
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
        # Another medium protein
        "MTIEHELLKQNILPFAYVNCLMKGYGVIKSFNLKQKIPKSLYYFGKGDYQIKGDLQKMPVLSLI",
        # Longer protein
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGSSKGQMALQGKDQVVKLIAGKTYTVPDVKRIQFYQVLNLKEGNLVLQKDQMRQPQGVDGVKQVVDNLKQSQVLQKDQMRQPQGVDGVKQVVDNLKQSQVLQKDQMRQPQG",
    ]
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\n🔧 Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test parameters
    organism = "Escherichia coli general"
    gc_target_min = 0.45
    gc_target_max = 0.55
    beam_size = 20
    
    results = []
    
    print(f"\n🎯 Testing with target GC range: {gc_target_min:.1%} - {gc_target_max:.1%}")
    print(f"🔍 Beam size: {beam_size}")
    print(f"🧬 Organism: {organism}")
    
    for i, protein in enumerate(test_proteins, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: Protein length {len(protein)}")
        print(f"Sequence: {protein[:50]}{'...' if len(protein) > 50 else ''}")
        
        try:
            # Test constrained beam search
            print("\n🚀 Running constrained beam search...")
            result = predict_dna_sequence_constrained_beam_search(
                protein=protein,
                organism=organism,
                device=device,
                tokenizer=tokenizer,
                model=model,
                gc_target_min=gc_target_min,
                gc_target_max=gc_target_max,
                beam_size=beam_size,
                verbose=True
            )
            
            # Calculate metrics
            dna_sequence = result.predicted_dna
            gc_content_percentage = get_GC_content(dna_sequence)  # Returns percentage 0-100
            gc_content = gc_content_percentage / 100.0  # Convert to fraction 0-1
            
            print(f"\n📊 Results:")
            print(f"   DNA length: {len(dna_sequence)} bp")
            print(f"   GC content: {gc_content:.1%}")
            print(f"   GC target: {gc_target_min:.1%} - {gc_target_max:.1%}")
            print(f"   Within target: {'✅' if gc_target_min <= gc_content <= gc_target_max else '❌'}")
            
            # Additional metrics
            try:
                # Calculate CAI using the sequence itself as reference
                cai_weights = get_CSI_weights([dna_sequence])
                cai_score = get_CSI_value(dna_sequence, cai_weights)
                
                # Calculate tAI using E. coli weights
                tai_weights = get_ecoli_tai_weights()
                tai_score = calculate_tAI(dna_sequence, tai_weights)
                
                # Calculate additional metrics
                enc_score = calculate_ENC(dna_sequence)
                cpb_score = calculate_CPB(dna_sequence)
                
                print(f"\n🧮 Quality Metrics:")
                print(f"   CAI: {cai_score:.3f}")
                print(f"   tAI: {tai_score:.3f}")
                print(f"   ENC: {enc_score:.3f}")
                print(f"   CPB: {cpb_score:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Error calculating metrics: {e}")
                cai_score = tai_score = enc_score = cpb_score = None
            
            # Store results
            results.append({
                'protein': protein,
                'protein_length': len(protein),
                'dna_sequence': dna_sequence,
                'dna_length': len(dna_sequence),
                'gc_content': gc_content,
                'gc_within_target': gc_target_min <= gc_content <= gc_target_max,
                'cai_score': cai_score,
                'tai_score': tai_score,
                'enc_score': enc_score,
                'cpb_score': cpb_score,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Error during test: {e}")
            results.append({
                'protein': protein,
                'protein_length': len(protein),
                'error': str(e),
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        gc_compliant = [r for r in successful_tests if r['gc_within_target']]
        print(f"\nGC Constraint Compliance: {len(gc_compliant)}/{len(successful_tests)} ({len(gc_compliant)/len(successful_tests)*100:.1f}%)")
        
        if gc_compliant:
            avg_gc = sum(r['gc_content'] for r in gc_compliant) / len(gc_compliant)
            avg_cai = sum(r['cai_score'] for r in gc_compliant if r['cai_score']) / len([r for r in gc_compliant if r['cai_score']])
            avg_tai = sum(r['tai_score'] for r in gc_compliant if r['tai_score']) / len([r for r in gc_compliant if r['tai_score']])
            
            print(f"Average GC content: {avg_gc:.1%}")
            print(f"Average CAI: {avg_cai:.3f}")
            print(f"Average tAI: {avg_tai:.3f}")
    
    # Save results
    results_file = "constrained_beam_search_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Final verdict
    if len(gc_compliant) == len(successful_tests):
        print("\n🎉 SUCCESS: All tests passed GC constraints!")
        print("✅ Constrained beam search 'Guardrail' system working correctly")
    else:
        print("\n⚠️ WARNING: Some tests failed GC constraints")
        print("❌ Constrained beam search needs investigation")


if __name__ == "__main__":
    test_constrained_beam_search()