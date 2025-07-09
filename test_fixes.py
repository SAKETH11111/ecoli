#!/usr/bin/env python3
"""
Test script to validate Priority Fixes for Constrained Beam Search
=================================================================

This script tests the three priority fixes:
1. Exact per-residue GC bounds in feasibility test
2. Position-aware GC penalty schedule
3. Adaptive beam rescue when beam count < K

Focus: Test the 175aa protein that previously failed (55.5% GC violation)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence_constrained_beam_search
from CodonTransformer.CodonEvaluation import get_GC_content

def test_priority_fixes():
    """Test the priority fixes on the previously failing 175aa protein."""

    print("🧪 Testing Priority Fixes for Constrained Beam Search")
    print("=" * 60)

    # Load model and tokenizer
    print("📚 Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")

    # Test cases - including the problematic 175aa protein
    test_cases = [
        {
            "name": "Short protein (10aa)",
            "protein": "MKQHKAMIVA",
            "expected_result": "should_pass"
        },
        {
            "name": "Medium protein (61aa)",
            "protein": "MKQHKAMIVALHRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQW",
            "expected_result": "should_pass"
        },
        {
            "name": "Long protein (175aa) - Previously failed with 55.5% GC",
            "protein": "MKQHKAMIVALHRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQWQRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQWQRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQWQRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQWQRQGQHVNVRTHTGEKPFACEAIQW",
            "expected_result": "previously_failed"
        }
    ]

    print(f"🎯 Target GC range: 45.0% - 55.0%")
    print(f"🔧 Priority fixes implemented:")
    print(f"   ✅ Fix #1: Exact per-residue GC bounds")
    print(f"   ✅ Fix #2: Position-aware GC penalty schedule")
    print(f"   ✅ Fix #3: Adaptive beam rescue")
    print()

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"   Length: {len(test_case['protein'])} amino acids")

        try:
            # Run constrained beam search with verbose output
            print(f"   🚀 Running enhanced constrained beam search...")

            result = predict_dna_sequence_constrained_beam_search(
                protein=test_case['protein'],
                organism="Escherichia coli general",
                device=device,
                tokenizer=tokenizer,
                model=model,
                gc_target_min=0.45,
                gc_target_max=0.55,
                beam_size=None,  # Adaptive
                match_protein=True,
                verbose=True,  # Enable verbose output
                gc_weight=0.5,
            )

            dna_sequence = result.predicted_dna
            gc_content = get_GC_content(dna_sequence)

            print(f"   ✅ SUCCESS!")
            print(f"   📊 Results:")
            print(f"      DNA length: {len(dna_sequence)} bp")
            print(f"      GC content: {gc_content:.1f}%")
            print(f"      Within target: {45.0 <= gc_content <= 55.0}")

            # Validate constraint satisfaction
            if 45.0 <= gc_content <= 55.0:
                print(f"      🎯 CONSTRAINT SATISFIED")
                if test_case['expected_result'] == 'previously_failed':
                    print(f"      🎉 FIXED! Previously failed with 55.5% GC")
            else:
                print(f"      ❌ CONSTRAINT VIOLATED")
                print(f"      ⚠️  GC content {gc_content:.1f}% outside target range")

        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            if test_case['expected_result'] == 'previously_failed':
                print(f"      🔄 Still needs work - this was the problematic case")

        print()

    print("🏁 Priority Fixes Test Complete")
    print("=" * 60)

def benchmark_performance():
    """Quick benchmark to measure performance improvements."""

    print("⚡ Performance Benchmark")
    print("-" * 30)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")

    # Test protein
    test_protein = "MKQHKAMIVALHRQGQHVNVRTHTGEKPFACEAIQQQRQGQHVNVRTHTGEKPFACEAIQW"

    print(f"Test protein: {len(test_protein)} amino acids")
    print(f"Device: {device}")

    import time
    start_time = time.time()

    try:
        result = predict_dna_sequence_constrained_beam_search(
            protein=test_protein,
            organism="Escherichia coli general",
            device=device,
            tokenizer=tokenizer,
            model=model,
            gc_target_min=0.45,
            gc_target_max=0.55,
            beam_size=None,  # Adaptive
            match_protein=True,
            verbose=False,  # Quiet for benchmark
            gc_weight=0.5,
        )

        end_time = time.time()
        elapsed = end_time - start_time

        dna_sequence = result.predicted_dna
        gc_content = get_GC_content(dna_sequence)

        print(f"✅ Success in {elapsed:.2f} seconds")
        print(f"   GC content: {gc_content:.1f}%")
        print(f"   Constraint satisfied: {45.0 <= gc_content <= 55.0}")

    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"❌ Failed after {elapsed:.2f} seconds: {str(e)}")

if __name__ == "__main__":
    # Run the tests
    test_priority_fixes()
    print()
    benchmark_performance()
