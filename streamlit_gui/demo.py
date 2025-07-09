#!/usr/bin/env python3
"""
Demo script for CodonTransformer Streamlit GUI

This script demonstrates the GUI functionality with example sequences
and showcases the key features of the CodonTransformer optimization tool.
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def print_header():
    """Print demo header"""
    print("🧬" * 20)
    print("  CodonTransformer GUI Demo")
    print("🧬" * 20)
    print()

def print_section(title):
    """Print section header"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def demo_validation():
    """Demonstrate protein sequence validation"""
    print_section("Protein Sequence Validation")

    # Import validation function
    from streamlit_gui.app import validate_protein_sequence

    test_sequences = [
        ("MKTVRQERLK", "Valid short peptide"),
        ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG", "Valid longer protein"),
        ("MKTVRQERLKX", "Invalid character (X)"),
        ("MK", "Too short"),
        ("mktvrqerlk", "Lowercase (should work)"),
        ("MKTVRQERLK*", "With stop codon"),
    ]

    for seq, description in test_sequences:
        is_valid, message = validate_protein_sequence(seq)
        status = "✅" if is_valid else "❌"
        print(f"{status} {description}: {message}")

def demo_metrics():
    """Demonstrate metrics calculation"""
    print_section("Metrics Calculation Demo")

    from streamlit_gui.app import calculate_input_metrics

    example_proteins = [
        ("MKTVRQERLK", "Short peptide (10 AA)"),
        ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG", "Medium protein (67 AA)"),
        ("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTE", "Long protein (72 AA)"),
    ]

    organism = "Escherichia coli general"

    for protein, description in example_proteins:
        print(f"\n🧪 {description}")
        print(f"   Sequence: {protein[:30]}{'...' if len(protein) > 30 else ''}")

        metrics = calculate_input_metrics(protein, organism)

        print(f"   📏 Length: {metrics['length']} amino acids")
        print(f"   🧬 GC Content: {metrics['gc_content']:.1f}%")
        if metrics['tai']:
            print(f"   📊 tAI: {metrics['tai']:.3f}")
        if metrics['cai']:
            print(f"   📊 CAI: {metrics['cai']:.3f}")
        else:
            print(f"   📊 CAI: Not available for this organism")

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print_section("Visualization Demo")

    from streamlit_gui.app import create_gc_content_plot, create_metrics_comparison_chart

    # Test DNA sequence for GC content plot
    test_dna = "ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC"

    print("🎨 Creating GC content sliding window plot...")
    try:
        fig = create_gc_content_plot(test_dna)
        print("   ✅ GC content plot created successfully")
        print(f"   📊 Analyzing {len(test_dna)} base pairs")
    except Exception as e:
        print(f"   ❌ Error creating GC plot: {e}")

    print("\n🎨 Creating metrics comparison chart...")
    try:
        before_metrics = {
            'gc_content': 45.2,
            'cai': 0.485,
            'tai': 0.312
        }
        after_metrics = {
            'gc_content': 52.1,
            'cai': 0.634,
            'tai': 0.456
        }
        fig = create_metrics_comparison_chart(before_metrics, after_metrics)
        print("   ✅ Comparison chart created successfully")
        print("   📈 Shows improvement in all metrics")
    except Exception as e:
        print(f"   ❌ Error creating comparison chart: {e}")

def demo_codon_evaluation():
    """Demonstrate CodonEvaluation functions"""
    print_section("CodonEvaluation Functions Demo")

    from CodonTransformer.CodonEvaluation import get_GC_content, calculate_tAI, get_ecoli_tai_weights

    test_sequences = [
        ("ATGGCGAAAGCGCTGTATCGC", "High GC content"),
        ("ATGAAATTTATTTATTATTAT", "Low GC content"),
        ("ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC", "Medium length"),
    ]

    print("🧬 Testing GC content calculation:")
    for seq, description in test_sequences:
        gc_content = get_GC_content(seq)
        print(f"   {description}: {gc_content:.1f}%")

    print("\n🧬 Testing tAI calculation:")
    try:
        tai_weights = get_ecoli_tai_weights()
        for seq, description in test_sequences:
            tai_value = calculate_tAI(seq, tai_weights)
            print(f"   {description}: {tai_value:.3f}")
    except Exception as e:
        print(f"   ❌ tAI calculation error: {e}")

def demo_model_info():
    """Show model information"""
    print_section("Model Information")

    try:
        import torch
        from transformers import AutoTokenizer

        print("🤖 Model Details:")
        print("   📦 Model: adibvafa/CodonTransformer")
        print("   🏗️  Architecture: BigBird Transformer")
        print("   📊 Task: Masked Language Modeling for codon optimization")

        print(f"\n💻 System Information:")
        print(f"   🔧 PyTorch: {torch.__version__}")
        print(f"   🎯 Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        print(f"\n🔤 Tokenizer Test:")
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        print(f"   ✅ Tokenizer loaded: {len(tokenizer)} tokens")
        print(f"   📝 Vocab size: {tokenizer.vocab_size}")

    except Exception as e:
        print(f"   ❌ Error loading model info: {e}")

def demo_gui_features():
    """Show GUI features overview"""
    print_section("GUI Features Overview")

    features = [
        ("🔍 Real-time Validation", "Instant feedback on protein sequence validity"),
        ("📊 Metrics Dashboard", "GC content, CAI, tAI calculations"),
        ("🎯 Constrained Optimization", "GC content control with beam search"),
        ("📈 Visual Analytics", "Interactive plots and comparisons"),
        ("⚙️ Configurable Parameters", "Organism selection, beam size, GC targets"),
        ("💾 Export Options", "Download optimized sequences"),
        ("🔄 Progress Tracking", "Real-time optimization progress"),
        ("📱 Responsive Design", "Works on desktop and mobile"),
    ]

    for feature, description in features:
        print(f"   {feature}: {description}")

def demo_usage_examples():
    """Show usage examples"""
    print_section("Usage Examples")

    examples = [
        {
            "name": "Short Peptide Optimization",
            "protein": "MKTVRQERLK",
            "organism": "Escherichia coli general",
            "use_case": "Quick testing and validation"
        },
        {
            "name": "Insulin Chain A",
            "protein": "GIVEQCCTSICSLYQLENYCN",
            "organism": "Escherichia coli general",
            "use_case": "Pharmaceutical protein production"
        },
        {
            "name": "Green Fluorescent Protein (partial)",
            "protein": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC",
            "organism": "Escherichia coli general",
            "use_case": "Research marker protein"
        },
        {
            "name": "Yeast Expression",
            "protein": "MKTVRQERLKSIVRILERSKEPVSGAQ",
            "organism": "Saccharomyces cerevisiae",
            "use_case": "Eukaryotic protein expression"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n🧪 Example {i}: {example['name']}")
        print(f"   Protein: {example['protein'][:40]}{'...' if len(example['protein']) > 40 else ''}")
        print(f"   Organism: {example['organism']}")
        print(f"   Use case: {example['use_case']}")
        print(f"   Length: {len(example['protein'])} amino acids")

def demo_launch_instructions():
    """Show how to launch the GUI"""
    print_section("How to Launch the GUI")

    print("🚀 Launch Options:")
    print()
    print("   Option 1 - Using the launcher script:")
    print("   $ cd ecoli/streamlit_gui")
    print("   $ python run_gui.py")
    print()
    print("   Option 2 - Direct streamlit command:")
    print("   $ cd ecoli/streamlit_gui")
    print("   $ source ../codon_env/bin/activate")
    print("   $ streamlit run app.py")
    print()
    print("   Option 3 - With custom port:")
    print("   $ streamlit run app.py --server.port 8502")
    print()
    print("📱 Access the GUI:")
    print("   🌐 Web browser: http://localhost:8501")
    print("   📋 The GUI will automatically open in your default browser")
    print()
    print("⚡ Performance Tips:")
    print("   • Use GPU if available for faster processing")
    print("   • Start with shorter sequences for testing")
    print("   • Adjust beam size based on sequence length")
    print("   • Close other applications to free up memory")

def main():
    """Run the complete demo"""
    print_header()

    print("This demo showcases the CodonTransformer Streamlit GUI capabilities.")
    print("The GUI provides an intuitive interface for protein codon optimization.")
    print()

    try:
        demo_validation()
        demo_metrics()
        demo_visualization()
        demo_codon_evaluation()
        demo_model_info()
        demo_gui_features()
        demo_usage_examples()
        demo_launch_instructions()

        print("\n🎉 Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Launch the GUI using one of the methods above")
        print("2. Try the example sequences provided")
        print("3. Experiment with different organisms and settings")
        print("4. Compare optimization results")
        print()
        print("Happy optimizing! 🧬✨")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
