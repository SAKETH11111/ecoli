import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, BigBirdForMaskedLM
import time
import threading
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Import CodonTransformer modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CodonTransformer.CodonPrediction import (
    predict_dna_sequence,
    predict_dna_sequence_constrained_beam_search
)
from CodonTransformer.CodonEvaluation import (
    get_GC_content,
    calculate_tAI,
    get_ecoli_tai_weights,
    get_CSI_weights,
    get_CSI_value
)
from CodonTransformer.CodonUtils import get_organism2id_dict

# Page configuration
st.set_page_config(
    page_title="CodonTransformer GUI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False
if 'results' not in st.session_state:
    st.session_state.results = None

def load_model_and_tokenizer():
    """Load the model and tokenizer with progress tracking"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading CodonTransformer model... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading tokenizer...")
            progress_bar.progress(25)
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

            status_text.text("Loading model...")
            progress_bar.progress(50)
            st.session_state.model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")

            status_text.text("Moving model to device...")
            progress_bar.progress(75)
            st.session_state.model = st.session_state.model.to(st.session_state.device)

            status_text.text("Model loaded successfully!")
            progress_bar.progress(100)
            time.sleep(0.5)

            status_text.empty()
            progress_bar.empty()

def validate_protein_sequence(protein: str) -> Tuple[bool, str]:
    """Validate protein sequence and return status and message"""
    if not protein:
        return False, "Protein sequence cannot be empty"

    # Remove whitespace and convert to uppercase
    protein = protein.strip().upper()

    # Check for valid amino acids
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY*_")
    invalid_chars = set(protein) - valid_amino_acids

    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(invalid_chars)}"

    # Check minimum length
    if len(protein) < 3:
        return False, "Protein sequence must be at least 3 amino acids long"

    # Check maximum length (for performance)
    if len(protein) > 500:
        return False, "Protein sequence too long (max 500 amino acids for this demo)"

    return True, "Valid protein sequence"

def calculate_input_metrics(protein: str, organism: str) -> Dict:
    """Calculate metrics for the input protein sequence"""
    # For input analysis, we'll use the most frequent codon for each amino acid
    # This gives us a baseline to compare against

    # Simple codon frequency for E. coli (most common codons)
    most_frequent_codons = {
        'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
        'G': 'GGC', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTG',
        'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG', 'R': 'CGC',
        'S': 'TCG', 'T': 'ACG', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT',
        '*': 'TAA', '_': 'TAA'
    }

    # Generate baseline DNA sequence
    baseline_dna = ''.join([most_frequent_codons.get(aa, 'NNN') for aa in protein])

    # Calculate metrics
    metrics = {
        'length': len(protein),
        'gc_content': get_GC_content(baseline_dna),
        'baseline_dna': baseline_dna
    }

    # Calculate CAI if possible
    try:
        csi_weights = get_CSI_weights()
        if organism.lower() in csi_weights:
            metrics['cai'] = get_CSI_value(baseline_dna, csi_weights[organism.lower()])
        else:
            metrics['cai'] = None
    except:
        metrics['cai'] = None

    # Calculate tAI
    try:
        tai_weights = get_ecoli_tai_weights()
        metrics['tai'] = calculate_tAI(baseline_dna, tai_weights)
    except:
        metrics['tai'] = None

    return metrics

def create_gc_content_plot(sequence: str, window_size: int = 50) -> go.Figure:
    """Create a sliding window GC content plot"""
    if len(sequence) < window_size:
        window_size = len(sequence) // 3

    positions = []
    gc_values = []

    for i in range(0, len(sequence) - window_size + 1, 3):  # Step by codons
        window = sequence[i:i + window_size]
        gc_content = get_GC_content(window)
        positions.append(i // 3)  # Position in codons
        gc_values.append(gc_content)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions,
        y=gc_values,
        mode='lines',
        name='GC Content',
        line=dict(color='blue', width=2)
    ))

    # Add target range
    fig.add_hline(y=45, line_dash="dash", line_color="red",
                  annotation_text="Min Target (45%)")
    fig.add_hline(y=55, line_dash="dash", line_color="red",
                  annotation_text="Max Target (55%)")

    fig.update_layout(
        title=f'GC Content (sliding window: {window_size} bp)',
        xaxis_title='Position (codons)',
        yaxis_title='GC Content (%)',
        height=300
    )

    return fig

def create_metrics_comparison_chart(before_metrics: Dict, after_metrics: Dict) -> go.Figure:
    """Create a comparison chart for metrics"""
    metrics_names = ['GC Content (%)', 'CAI', 'tAI']
    before_values = [
        before_metrics.get('gc_content', 0),
        before_metrics.get('cai', 0) if before_metrics.get('cai') else 0,
        before_metrics.get('tai', 0) if before_metrics.get('tai') else 0
    ]
    after_values = [
        after_metrics.get('gc_content', 0),
        after_metrics.get('cai', 0) if after_metrics.get('cai') else 0,
        after_metrics.get('tai', 0) if after_metrics.get('tai') else 0
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before Optimization',
        x=metrics_names,
        y=before_values,
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='After Optimization',
        x=metrics_names,
        y=after_values,
        marker_color='darkblue'
    ))

    fig.update_layout(
        title='Metrics Comparison: Before vs After',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=400
    )

    return fig

def run_optimization(protein: str, organism: str, use_constrained: bool,
                    gc_min: float, gc_max: float, beam_size: int):
    """Run the optimization in a separate thread"""
    st.session_state.optimization_running = True

    try:
        if use_constrained:
            result = predict_dna_sequence_constrained_beam_search(
                protein=protein,
                organism=organism,
                device=st.session_state.device,
                tokenizer=st.session_state.tokenizer,
                model=st.session_state.model,
                gc_target_min=gc_min,
                gc_target_max=gc_max,
                beam_size=beam_size,
                verbose=True
            )
        else:
            result = predict_dna_sequence(
                protein=protein,
                organism=organism,
                device=st.session_state.device,
                tokenizer=st.session_state.tokenizer,
                model=st.session_state.model,
                deterministic=True
            )

        st.session_state.results = result

    except Exception as e:
        st.session_state.results = f"Error: {str(e)}"

    finally:
        st.session_state.optimization_running = False

def main():
    st.title("🧬 CodonTransformer GUI")
    st.markdown("**Optimize protein sequences for codon usage with real-time analysis**")

    # Load model
    load_model_and_tokenizer()

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Organism selection
    organism_options = [
        "Escherichia coli general",
        "Saccharomyces cerevisiae",
        "Homo sapiens",
        "Bacillus subtilis",
        "Pichia pastoris"
    ]
    organism = st.sidebar.selectbox("Select Organism", organism_options)

    # Optimization method
    use_constrained = st.sidebar.checkbox(
        "Use Constrained Beam Search",
        value=True,
        help="Use the improved constrained beam search with GC content control"
    )

    if use_constrained:
        st.sidebar.subheader("GC Content Control")
        gc_min = st.sidebar.slider("Min GC Content (%)", 30, 70, 45) / 100
        gc_max = st.sidebar.slider("Max GC Content (%)", 30, 70, 55) / 100
        beam_size = st.sidebar.slider("Beam Size", 5, 50, 20)
    else:
        gc_min, gc_max, beam_size = 0.45, 0.55, 20

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")

        # Protein sequence input
        protein_input = st.text_area(
            "Enter Protein Sequence",
            height=150,
            placeholder="Enter single-letter amino acid sequence (e.g., MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA"
        )

        # Validate input
        if protein_input:
            is_valid, message = validate_protein_sequence(protein_input)
            if is_valid:
                st.success(f"✅ {message}")
                protein_clean = protein_input.strip().upper()

                # Calculate input metrics
                input_metrics = calculate_input_metrics(protein_clean, organism)

                st.subheader("Input Analysis")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    st.metric("Length", f"{input_metrics['length']} AA")
                    st.metric("GC Content", f"{input_metrics['gc_content']:.1f}%")

                with metrics_col2:
                    if input_metrics['cai']:
                        st.metric("CAI (baseline)", f"{input_metrics['cai']:.3f}")
                    else:
                        st.metric("CAI", "N/A")

                with metrics_col3:
                    if input_metrics['tai']:
                        st.metric("tAI (baseline)", f"{input_metrics['tai']:.3f}")
                    else:
                        st.metric("tAI", "N/A")

                # GC content plot for input
                if len(input_metrics['baseline_dna']) > 150:
                    fig = create_gc_content_plot(input_metrics['baseline_dna'])
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ {message}")
                protein_clean = None
        else:
            protein_clean = None

    with col2:
        st.header("Optimization Results")

        # Optimization button
        if protein_clean and not st.session_state.optimization_running:
            if st.button("🚀 Optimize Sequence", type="primary"):
                st.session_state.results = None
                # Run optimization
                run_optimization(protein_clean, organism, use_constrained,
                               gc_min, gc_max, beam_size)

        # Show progress
        if st.session_state.optimization_running:
            st.info("🔄 Optimizing sequence... This may take a few moments.")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate progress (in real implementation, this would come from the model)
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("Initializing beam search...")
                elif i < 60:
                    status_text.text("Exploring codon combinations...")
                elif i < 90:
                    status_text.text("Optimizing GC content...")
                else:
                    status_text.text("Finalizing results...")
                time.sleep(0.1)

            progress_bar.empty()
            status_text.empty()

        # Display results
        if st.session_state.results and not st.session_state.optimization_running:
            if isinstance(st.session_state.results, str):
                st.error(st.session_state.results)
            else:
                result = st.session_state.results

                # Calculate optimized metrics
                optimized_metrics = {
                    'gc_content': get_GC_content(result.predicted_dna),
                    'length': len(result.predicted_dna)
                }

                # Calculate CAI and tAI for optimized sequence
                try:
                    csi_weights = get_CSI_weights()
                    if organism.lower() in csi_weights:
                        optimized_metrics['cai'] = get_CSI_value(result.predicted_dna, csi_weights[organism.lower()])
                    else:
                        optimized_metrics['cai'] = None
                except:
                    optimized_metrics['cai'] = None

                try:
                    tai_weights = get_ecoli_tai_weights()
                    optimized_metrics['tai'] = calculate_tAI(result.predicted_dna, tai_weights)
                except:
                    optimized_metrics['tai'] = None

                # Display metrics
                st.subheader("Optimized Metrics")
                opt_col1, opt_col2, opt_col3 = st.columns(3)

                with opt_col1:
                    st.metric("GC Content", f"{optimized_metrics['gc_content']:.1f}%")

                with opt_col2:
                    if optimized_metrics['cai']:
                        st.metric("CAI", f"{optimized_metrics['cai']:.3f}")
                    else:
                        st.metric("CAI", "N/A")

                with opt_col3:
                    if optimized_metrics['tai']:
                        st.metric("tAI", f"{optimized_metrics['tai']:.3f}")
                    else:
                        st.metric("tAI", "N/A")

                # DNA sequence display
                st.subheader("Optimized DNA Sequence")
                st.text_area("DNA Sequence", result.predicted_dna, height=100)

                # Download button
                st.download_button(
                    label="📥 Download DNA Sequence",
                    data=result.predicted_dna,
                    file_name=f"optimized_sequence_{organism.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

    # Full-width comparison section
    if (st.session_state.results and
        not st.session_state.optimization_running and
        protein_clean and
        not isinstance(st.session_state.results, str)):

        st.header("📊 Before vs After Comparison")

        # Calculate metrics for comparison
        input_metrics = calculate_input_metrics(protein_clean, organism)
        result = st.session_state.results

        optimized_metrics = {
            'gc_content': get_GC_content(result.predicted_dna),
        }

        # Add CAI and tAI
        try:
            csi_weights = get_CSI_weights()
            if organism.lower() in csi_weights:
                optimized_metrics['cai'] = get_CSI_value(result.predicted_dna, csi_weights[organism.lower()])
            else:
                optimized_metrics['cai'] = None
        except:
            optimized_metrics['cai'] = None

        try:
            tai_weights = get_ecoli_tai_weights()
            optimized_metrics['tai'] = calculate_tAI(result.predicted_dna, tai_weights)
        except:
            optimized_metrics['tai'] = None

        # Comparison chart
        if input_metrics['cai'] and optimized_metrics['cai']:
            comp_fig = create_metrics_comparison_chart(input_metrics, optimized_metrics)
            st.plotly_chart(comp_fig, use_container_width=True)

        # Side-by-side GC content plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before Optimization")
            if len(input_metrics['baseline_dna']) > 150:
                fig_before = create_gc_content_plot(input_metrics['baseline_dna'])
                st.plotly_chart(fig_before, use_container_width=True)

        with col2:
            st.subheader("After Optimization")
            if len(result.predicted_dna) > 150:
                fig_after = create_gc_content_plot(result.predicted_dna)
                st.plotly_chart(fig_after, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**CodonTransformer GUI** - Optimize your protein sequences with AI-powered codon usage optimization")
    st.markdown("Built with Streamlit 🎈")

if __name__ == "__main__":
    main()
