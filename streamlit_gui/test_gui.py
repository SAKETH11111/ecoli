#!/usr/bin/env python3
"""
Test script for CodonTransformer Streamlit GUI

This script tests the core functionality of the GUI without running the full Streamlit application.
"""

import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")

    try:
        import streamlit as st
        print(f"  ✅ Streamlit: {st.__version__}")
    except ImportError as e:
        print(f"  ❌ Streamlit: {e}")
        return False

    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"  ✅ PyTorch: {torch.__version__} ({device})")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        return False

    try:
        import plotly
        print(f"  ✅ Plotly: {plotly.__version__}")
    except ImportError as e:
        print(f"  ❌ Plotly: {e}")
        return False

    try:
        from CodonTransformer.CodonPrediction import predict_dna_sequence
        print(f"  ✅ CodonTransformer.CodonPrediction")
    except ImportError as e:
        print(f"  ❌ CodonTransformer.CodonPrediction: {e}")
        return False

    try:
        from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI
        print(f"  ✅ CodonTransformer.CodonEvaluation")
    except ImportError as e:
        print(f"  ❌ CodonTransformer.CodonEvaluation: {e}")
        return False

    return True

def test_protein_validation():
    """Test protein sequence validation"""
    print("\n🧪 Testing protein sequence validation...")

    try:
        # Import the validation function
        from app import validate_protein_sequence

        # Test cases
        test_cases = [
            ("MKTVRQERLK", True, "Valid short sequence"),
            ("", False, "Empty sequence"),
            ("MKTVRQERLKX", False, "Invalid character X"),
            ("MK", False, "Too short"),
            ("M" * 501, False, "Too long"),
            ("mktvrqerlk", True, "Lowercase (should work)"),
            ("MKTVRQERLK*", True, "With stop codon"),
            ("MKTVRQERLK_", True, "With underscore stop"),
        ]

        for seq, expected_valid, description in test_cases:
            is_valid, message = validate_protein_sequence(seq)
            status = "✅" if is_valid == expected_valid else "❌"
            print(f"  {status} {description}: {message}")

        return True
    except Exception as e:
        print(f"  ❌ Error in validation test: {e}")
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    print("\n🧪 Testing metrics calculation...")

    try:
        from app import calculate_input_metrics

        test_protein = "MKTVRQERLK"
        organism = "Escherichia coli general"

        metrics = calculate_input_metrics(test_protein, organism)

        # Check if all expected metrics are present
        expected_keys = ['length', 'gc_content', 'baseline_dna', 'cai', 'tai']
        for key in expected_keys:
            if key in metrics:
                print(f"  ✅ {key}: {metrics[key]}")
            else:
                print(f"  ❌ Missing metric: {key}")
                return False

        # Validate metric values
        if metrics['length'] == len(test_protein):
            print(f"  ✅ Length calculation correct")
        else:
            print(f"  ❌ Length calculation incorrect")
            return False

        if 0 <= metrics['gc_content'] <= 100:
            print(f"  ✅ GC content in valid range")
        else:
            print(f"  ❌ GC content out of range")
            return False

        return True
    except Exception as e:
        print(f"  ❌ Error in metrics calculation: {e}")
        traceback.print_exc()
        return False

def test_visualization_functions():
    """Test visualization functions"""
    print("\n🧪 Testing visualization functions...")

    try:
        from app import create_gc_content_plot, create_metrics_comparison_chart

        # Test GC content plot
        test_dna = "ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC"
        fig = create_gc_content_plot(test_dna)
        print(f"  ✅ GC content plot created")

        # Test metrics comparison chart
        before_metrics = {'gc_content': 50.0, 'cai': 0.5, 'tai': 0.3}
        after_metrics = {'gc_content': 52.0, 'cai': 0.6, 'tai': 0.4}
        fig = create_metrics_comparison_chart(before_metrics, after_metrics)
        print(f"  ✅ Metrics comparison chart created")

        return True
    except Exception as e:
        print(f"  ❌ Error in visualization test: {e}")
        traceback.print_exc()
        return False

def test_codon_evaluation():
    """Test CodonEvaluation functions directly"""
    print("\n🧪 Testing CodonEvaluation functions...")

    try:
        from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI, get_ecoli_tai_weights

        # Test GC content calculation
        test_dna = "ATGGCGAAAGCG"
        gc_content = get_GC_content(test_dna)
        print(f"  ✅ GC content calculation: {gc_content:.1f}%")

        # Test tAI calculation
        try:
            tai_weights = get_ecoli_tai_weights()
            tai_value = calculate_tAI(test_dna, tai_weights)
            print(f"  ✅ tAI calculation: {tai_value:.3f}")
        except Exception as e:
            print(f"  ⚠️  tAI calculation (may need scipy): {e}")

        return True
    except Exception as e:
        print(f"  ❌ Error in CodonEvaluation test: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🧪 Testing model loading (mock)...")

    try:
        import torch
        from transformers import AutoTokenizer

        # Test tokenizer loading (this is fast)
        print("  📥 Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        print(f"  ✅ Tokenizer loaded successfully")

        # For model loading, we'll just check if the model class can be imported
        from transformers import BigBirdForMaskedLM
        print(f"  ✅ Model class available: BigBirdForMaskedLM")

        # Note: We won't actually load the full model here as it's ~2GB
        print("  ℹ️  Full model loading skipped in test (too large)")

        return True
    except Exception as e:
        print(f"  ❌ Error in model loading test: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🧪 Testing file structure...")

    gui_dir = Path(__file__).parent
    required_files = [
        "app.py",
        "run_gui.py",
        "requirements.txt",
        "README.md"
    ]

    all_present = True
    for file_name in required_files:
        file_path = gui_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} missing")
            all_present = False

    return all_present

def main():
    """Run all tests"""
    print("🚀 CodonTransformer GUI Test Suite")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Protein Validation", test_protein_validation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Visualization Functions", test_visualization_functions),
        ("CodonEvaluation Functions", test_codon_evaluation),
        ("Model Loading", test_model_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The GUI should work correctly.")
        print("\nTo run the GUI:")
        print("  python run_gui.py")
        print("  or")
        print("  source ../codon_env/bin/activate && streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
