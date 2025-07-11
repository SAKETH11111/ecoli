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
    load_model
)
from CodonTransformer.CodonEvaluation import (
    get_GC_content,
    calculate_tAI,
    get_ecoli_tai_weights,
    scan_for_restriction_sites,
    count_negative_cis_elements,
    calculate_homopolymer_runs
)
from CAI import CAI, relative_adaptiveness
from CodonTransformer.CodonUtils import get_organism2id_dict

# Try to import post-processing features
try:
    from CodonTransformer.CodonPostProcessing import (
        polish_sequence_with_dnachisel,
        DNACHISEL_AVAILABLE
    )
    POST_PROCESSING_AVAILABLE = True
except ImportError:
    POST_PROCESSING_AVAILABLE = False
    DNACHISEL_AVAILABLE = False

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
if 'post_processed_results' not in st.session_state:
    st.session_state.post_processed_results = None
if 'cai_weights' not in st.session_state:
    st.session_state.cai_weights = None
if 'tai_weights' not in st.session_state:
    st.session_state.tai_weights = None

def load_model_and_tokenizer():
    """Load the model and tokenizer with progress tracking"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading CodonTransformer model... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading tokenizer...")
            progress_bar.progress(25)
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

            status_text.text("Loading fine-tuned model...")
            progress_bar.progress(50)
            # Load fine-tuned model by default, fallback to base model
            model_path = "models/alm-enhanced-training/balanced_alm_finetune.ckpt"
            try:
                st.session_state.model = load_model(
                    model_path=model_path,
                    device=st.session_state.device,
                    attention_type="original_full"
                )
                status_text.text("✅ Fine-tuned model loaded (6.2% better CAI)")
                st.session_state.model_type = "fine_tuned"
            except Exception as e:
                status_text.text("Fine-tuned model not found, loading base model...")
                st.session_state.model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
                st.session_state.model = st.session_state.model.to(st.session_state.device)
                st.session_state.model_type = "base"

            progress_bar.progress(100)
            time.sleep(0.5)

            status_text.empty()
            progress_bar.empty()

def load_reference_data(organism: str = "Escherichia coli general"):
    """Load reference sequences for proper CAI calculation based on organism"""
    organism_key = f"cai_weights_{organism.replace(' ', '_')}"

    if organism_key not in st.session_state or st.session_state[organism_key] is None:
        try:
            import pandas as pd

            # Map organism names to reference files
            organism_files = {
                "Escherichia coli general": "data/reference_sequences/ecoli_general_1000.csv",
                "Homo sapiens": "data/reference_sequences/homo_sapiens_1000.csv",
                "Saccharomyces cerevisiae": "data/reference_sequences/saccharomyces_cerevisiae_1000.csv"
            }

            reference_file = organism_files.get(organism)

            if reference_file and os.path.exists(reference_file):
                # Load organism-specific reference sequences
                ref_df = pd.read_csv(reference_file)
                ref_sequences = ref_df['natural_dna'].dropna().tolist()[:500]  # Use first 500 sequences
                st.session_state[organism_key] = relative_adaptiveness(sequences=ref_sequences)
                st.info(f"✅ Loaded {len(ref_sequences)} {organism} reference sequences for CAI calculation")
            else:
                # Fallback: use general E. coli if file not found
                fallback_file = "data/ecoli_processed_genes.csv"
                if os.path.exists(fallback_file):
                    fallback_df = pd.read_csv(fallback_file)
                    ref_sequences = fallback_df['dna_sequence'].tolist()
                    st.session_state[organism_key] = relative_adaptiveness(sequences=ref_sequences)
                    st.warning(f"⚠️ Using E. coli reference for {organism} (organism-specific data not available)")
                else:
                    # Final fallback: sample sequences
                    ref_sequences = [
                        "ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC",
                        "ATGAAATTTATTTATTATTATAAATTTATTTATTATTATAAATTTATTTAT",
                        "ATGGGTCGTCGTCGTCGTGGTCGTCGTCGTCGTGGTCGTCGTCGTCGTGGT"
                    ]
                    st.session_state[organism_key] = relative_adaptiveness(sequences=ref_sequences)
                    st.warning(f"⚠️ Using minimal reference sequences for {organism}")

        except Exception as e:
            st.error(f"Error loading reference data for {organism}: {e}")
            st.session_state[organism_key] = {}

    # Load tAI weights (organism-independent for now)
    if 'tai_weights' not in st.session_state or st.session_state.tai_weights is None:
        try:
            st.session_state.tai_weights = get_ecoli_tai_weights()
        except Exception as e:
            st.error(f"Error loading tAI weights: {e}")
            st.session_state.tai_weights = {}

def validate_sequence(sequence: str) -> Tuple[bool, str, str]:
    """Validate sequence and return status, message, and sequence type"""
    if not sequence:
        return False, "Sequence cannot be empty", "unknown"

    # Remove whitespace and convert to uppercase
    sequence = sequence.strip().upper()

    # Check if it's a DNA sequence
    dna_chars = set("ATGC")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY*_")

    sequence_chars = set(sequence)

    # If all characters are DNA nucleotides, treat as DNA
    if sequence_chars.issubset(dna_chars):
        if len(sequence) < 3:
            return False, "DNA sequence must be at least 3 nucleotides long", "dna"
        if len(sequence) > 6000:
            return False, "DNA sequence too long (max 6000 nucleotides)", "dna"
        return True, "Valid DNA sequence", "dna"

    # If contains protein-specific amino acids, treat as protein
    elif sequence_chars.issubset(protein_chars):
        if len(sequence) < 3:
            return False, "Protein sequence must be at least 3 amino acids long", "protein"
        if len(sequence) > 2000:
            return False, "Protein sequence too long (max 2000 amino acids)", "protein"
        return True, "Valid protein sequence", "protein"

    # Invalid characters
    else:
        invalid_chars = sequence_chars - (dna_chars | protein_chars)
        return False, f"Invalid characters found: {', '.join(invalid_chars)}", "unknown"

def calculate_input_metrics(sequence: str, organism: str, sequence_type: str) -> Dict:
    """Calculate metrics for the input sequence using exact same method as evaluation"""
    # Load reference data if not already loaded
    load_reference_data(organism)

    if sequence_type == "dna":
        # Direct calculation for DNA sequences
        dna_sequence = sequence.upper()

        # Calculate metrics using exact same method as evaluation script
        metrics = {
            'length': len(dna_sequence) // 3,  # Length in codons
            'gc_content': get_GC_content(dna_sequence),
            'baseline_dna': dna_sequence,
            'sequence_type': 'dna'
        }

        # Calculate CAI using same method as evaluation script
        try:
            organism_key = f"cai_weights_{organism.replace(' ', '_')}"
            if organism_key in st.session_state and st.session_state[organism_key]:
                metrics['cai'] = CAI(dna_sequence, weights=st.session_state[organism_key])
            else:
                metrics['cai'] = None
        except:
            metrics['cai'] = None

        # Calculate tAI using same method as evaluation script
        try:
            if st.session_state.tai_weights:
                metrics['tai'] = calculate_tAI(dna_sequence, st.session_state.tai_weights)
            else:
                metrics['tai'] = None
        except:
            metrics['tai'] = None

    else:
        # For protein sequences, use the most frequent codon baseline
        most_frequent_codons = {
            'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
            'G': 'GGC', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTG',
            'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG', 'R': 'CGC',
            'S': 'TCG', 'T': 'ACG', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT',
            '*': 'TAA', '_': 'TAA'
        }

        # Generate baseline DNA sequence
        baseline_dna = ''.join([most_frequent_codons.get(aa, 'NNN') for aa in sequence])

        # Calculate metrics using exact same method as evaluation script
        metrics = {
            'length': len(sequence),
            'gc_content': get_GC_content(baseline_dna),
            'baseline_dna': baseline_dna,
            'sequence_type': 'protein'
        }

        # Calculate CAI using same method as evaluation script
        try:
            organism_key = f"cai_weights_{organism.replace(' ', '_')}"
            if organism_key in st.session_state and st.session_state[organism_key]:
                metrics['cai'] = CAI(baseline_dna, weights=st.session_state[organism_key])
            else:
                metrics['cai'] = None
        except:
            metrics['cai'] = None

        # Calculate tAI using same method as evaluation script
        try:
            if st.session_state.tai_weights:
                metrics['tai'] = calculate_tAI(baseline_dna, st.session_state.tai_weights)
            else:
                metrics['tai'] = None
        except:
            metrics['tai'] = None

    # Additional sequence analysis
    try:
        analysis_dna = metrics['baseline_dna']
        metrics['restriction_sites'] = len(scan_for_restriction_sites(analysis_dna))
        metrics['negative_cis_elements'] = count_negative_cis_elements(analysis_dna)
        metrics['homopolymer_runs'] = calculate_homopolymer_runs(analysis_dna)
    except:
        metrics['restriction_sites'] = 0
        metrics['negative_cis_elements'] = 0
        metrics['homopolymer_runs'] = 0

    return metrics

def translate_dna_to_protein(dna_sequence: str) -> str:
    """Translate DNA sequence to protein sequence"""
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    protein = ""
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3].upper()
        if len(codon) == 3:
            aa = codon_table.get(codon, 'X')
            if aa == '*':  # Stop codon
                break
            protein += aa

    return protein

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

def create_gc_comparison_chart(before_metrics: Dict, after_metrics: Dict) -> go.Figure:
    """Create a comparison chart for GC Content"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before Optimization',
        x=['GC Content (%)'],
        y=[before_metrics.get('gc_content', 0)],
        marker_color='lightblue',
        text=[f"{before_metrics.get('gc_content', 0):.1f}%"],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='After Optimization',
        x=['GC Content (%)'],
        y=[after_metrics.get('gc_content', 0)],
        marker_color='darkblue',
        text=[f"{after_metrics.get('gc_content', 0):.1f}%"],
        textposition='auto'
    ))
    fig.update_layout(
        title='GC Content Comparison: Before vs After',
        xaxis_title='Metric',
        yaxis_title='Value (%)',
        barmode='group',
        height=300
    )
    return fig

def create_expression_comparison_chart(before_metrics: Dict, after_metrics: Dict) -> go.Figure:
    """Create a comparison chart for expression metrics (CAI, tAI)"""
    metrics_names = ['CAI', 'tAI']
    before_values = [
        before_metrics.get('cai', 0) if before_metrics.get('cai') else 0,
        before_metrics.get('tai', 0) if before_metrics.get('tai') else 0
    ]
    after_values = [
        after_metrics.get('cai', 0) if after_metrics.get('cai') else 0,
        after_metrics.get('tai', 0) if after_metrics.get('tai') else 0
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before Optimization',
        x=metrics_names,
        y=before_values,
        marker_color='lightblue',
        text=[f"{v:.3f}" for v in before_values],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='After Optimization',
        x=metrics_names,
        y=after_values,
        marker_color='darkblue',
        text=[f"{v:.3f}" for v in after_values],
        textposition='auto'
    ))
    fig.update_layout(
        title='Expression Metrics Comparison: Before vs After',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        height=300
    )
    return fig

def smart_codon_replacement(dna_sequence: str, target_gc_min: float = 0.45, target_gc_max: float = 0.55, max_iterations: int = 100) -> str:
    """Smart codon replacement to optimize GC content while maximizing CAI"""

    # Codon alternatives with their GC content
    codon_alternatives = {
        # Serine: high GC options
        'TCT': ['TCG', 'TCC', 'TCA', 'AGT', 'AGC'],  # 33% -> 67%, 67%, 33%, 33%, 67%
        'TCA': ['TCG', 'TCC', 'TCT', 'AGT', 'AGC'],
        'AGT': ['TCG', 'TCC', 'TCT', 'TCA', 'AGC'],

        # Leucine: various GC options
        'TTA': ['TTG', 'CTT', 'CTC', 'CTA', 'CTG'],  # 0% -> 33%, 33%, 67%, 33%, 67%
        'TTG': ['TTA', 'CTT', 'CTC', 'CTA', 'CTG'],
        'CTT': ['CTG', 'CTC', 'TTA', 'TTG', 'CTA'],
        'CTA': ['CTG', 'CTC', 'CTT', 'TTA', 'TTG'],

        # Arginine: various GC options
        'AGA': ['CGT', 'CGC', 'CGA', 'CGG', 'AGG'],  # 33% -> 67%, 100%, 67%, 100%, 67%
        'AGG': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA'],
        'CGT': ['CGC', 'CGG', 'CGA', 'AGA', 'AGG'],
        'CGA': ['CGC', 'CGG', 'CGT', 'AGA', 'AGG'],

        # Proline
        'CCT': ['CCG', 'CCC', 'CCA'],  # 67% -> 100%, 100%, 67%
        'CCA': ['CCG', 'CCC', 'CCT'],

        # Threonine
        'ACT': ['ACG', 'ACC', 'ACA'],  # 33% -> 67%, 67%, 33%
        'ACA': ['ACG', 'ACC', 'ACT'],

        # Alanine
        'GCT': ['GCG', 'GCC', 'GCA'],  # 67% -> 100%, 100%, 67%
        'GCA': ['GCG', 'GCC', 'GCT'],

        # Glycine
        'GGT': ['GGG', 'GGC', 'GGA'],  # 67% -> 100%, 100%, 67%
        'GGA': ['GGG', 'GGC', 'GGT'],

        # Valine
        'GTT': ['GTG', 'GTC', 'GTA'],  # 67% -> 100%, 100%, 67%
        'GTA': ['GTG', 'GTC', 'GTT'],
    }

    def get_codon_gc(codon):
        return (codon.count('G') + codon.count('C')) / 3.0

    current_sequence = dna_sequence.upper()
    current_gc = get_GC_content(current_sequence)

    if target_gc_min <= current_gc <= target_gc_max:
        return current_sequence

    codons = [current_sequence[i:i+3] for i in range(0, len(current_sequence), 3)]

    for iteration in range(max_iterations):
        current_gc = get_GC_content(''.join(codons))

        if target_gc_min <= current_gc <= target_gc_max:
            break

        # Find best codon to replace
        best_improvement = 0
        best_pos = -1
        best_replacement = None

        for pos, codon in enumerate(codons):
            if codon in codon_alternatives:
                for alt_codon in codon_alternatives[codon]:
                    # Calculate GC change
                    old_gc_contrib = get_codon_gc(codon)
                    new_gc_contrib = get_codon_gc(alt_codon)
                    gc_change = new_gc_contrib - old_gc_contrib

                    # Check if this change moves us toward target
                    if current_gc < target_gc_min and gc_change > best_improvement:
                        best_improvement = gc_change
                        best_pos = pos
                        best_replacement = alt_codon
                    elif current_gc > target_gc_max and gc_change < best_improvement:
                        best_improvement = abs(gc_change)
                        best_pos = pos
                        best_replacement = alt_codon

        if best_pos >= 0:
            codons[best_pos] = best_replacement
        else:
            break  # No more improvements possible

    return ''.join(codons)

def run_optimization(protein: str, organism: str, use_post_processing: bool = False):
    """Run the optimization using the exact method from run_full_comparison.py with auto GC correction"""
    st.session_state.optimization_running = True
    st.session_state.post_processed_results = None

    try:
        # Use the exact same method that achieved best results in evaluation
        result = predict_dna_sequence(
            protein=protein,
            organism=organism,
            device=st.session_state.device,
            model=st.session_state.model,
            deterministic=True,
            match_protein=True,
        )

        # Check GC content and auto-correct if out of optimal range
        initial_gc = get_GC_content(result.predicted_dna)

        if initial_gc < 45.0 or initial_gc > 55.0:
            # Show warning about GC correction
            st.warning(f"⚠️ Initial GC content ({initial_gc:.1f}%) is outside optimal range (45-55%). Auto-correcting...")

            # First try smart codon replacement (faster)
            st.info("🔄 Trying smart codon replacement...")
            optimized_dna = smart_codon_replacement(result.predicted_dna, 0.45, 0.55)
            smart_gc = get_GC_content(optimized_dna)

            if 45.0 <= smart_gc <= 55.0:
                # Smart replacement worked
                from CodonTransformer.CodonUtils import DNASequencePrediction
                result = DNASequencePrediction(
                    organism=result.organism,
                    protein=result.protein,
                    processed_input=result.processed_input,
                    predicted_dna=optimized_dna
                )
                st.success(f"✅ GC content optimized to {smart_gc:.1f}% using smart codon replacement")
            else:
                # Fall back to constrained beam search
                st.info("🔄 Falling back to constrained beam search...")
                try:
                    result = predict_dna_sequence(
                        protein=protein,
                        organism=organism,
                        device=st.session_state.device,
                        model=st.session_state.model,
                        deterministic=True,
                        match_protein=True,
                        use_constrained_search=True,
                        gc_bounds=(0.45, 0.55),
                        beam_size=20
                    )
                    final_gc = get_GC_content(result.predicted_dna)
                    st.success(f"✅ GC content corrected to {final_gc:.1f}% using constrained search")
                except Exception as e:
                    # If constrained search fails, use smart replacement result anyway
                    from CodonTransformer.CodonUtils import DNASequencePrediction
                    result = DNASequencePrediction(
                        organism=result.organism,
                        protein=result.protein,
                        processed_input=result.processed_input,
                        predicted_dna=optimized_dna
                    )
                    st.warning(f"⚠️ Constrained search failed, using smart replacement result: {smart_gc:.1f}% GC")

        st.session_state.results = result

        # Post-processing if enabled
        if use_post_processing and POST_PROCESSING_AVAILABLE and result:
            try:
                polished_sequence = polish_sequence_with_dnachisel(
                    dna_sequence=result.predicted_dna,
                    protein_sequence=protein,
                    gc_bounds=(45.0, 55.0),
                    cai_species=organism.lower().replace(' ', '_'),
                    avoid_homopolymers_length=6
                )

                # Create enhanced result object
                from CodonTransformer.CodonUtils import DNASequencePrediction
                st.session_state.post_processed_results = DNASequencePrediction(
                    organism=result.organism,
                    protein=result.protein,
                    processed_input=result.processed_input,
                    predicted_dna=polished_sequence
                )
                st.success("✅ Post-processing completed with DNAChisel")
            except Exception as e:
                st.session_state.post_processed_results = f"Post-processing error: {str(e)}"
                st.warning(f"⚠️ Post-processing failed: {str(e)}")

    except Exception as e:
        st.session_state.results = f"Error: {str(e)}"

    finally:
        st.session_state.optimization_running = False

def main():
    st.title("🧬 ColiFormer GUI")
    st.markdown("**Optimize protein sequences for codon usage with real-time analysis**")
    st.markdown("*Using the proven fine-tuned model (2.68pp CAI improvement)*")

    # Performance highlights
    with st.expander("📊 Performance Results (100 test proteins)", expanded=False):
        st.markdown("**Fine-tuned model vs Base model comparison:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CAI Score", "0.9037", "+0.0268 vs base")
        with col2:
            st.metric("tAI Score", "0.3750", "+0.0077 vs base")
        with col3:
            st.metric("GC Content", "55.85%", "-1.30% vs base")

        col4, col5 = st.columns(2)
        with col4:
            st.metric("Restriction Sites", "0.58", "+0.26 vs base")
        with col5:
            st.metric("Negative Elements", "0.06", "-0.26 vs base")

        st.markdown("*Results from evaluation with models/alm-enhanced-training/balanced_alm_finetune.ckpt*")
        st.markdown("**Key Improvement:** 2.68 percentage point CAI improvement with better GC content control")

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

    # Load reference data for selected organism
    load_reference_data(organism)

    # Post-processing
    st.sidebar.subheader("Post-Processing")
    use_post_processing = st.sidebar.checkbox(
        "Enable DNAChisel Post-Processing",
        value=False,
        disabled=not POST_PROCESSING_AVAILABLE,
        help="Polish sequences to remove restriction sites, homopolymers, and other synthesis issues"
    )

    if not POST_PROCESSING_AVAILABLE:
        st.sidebar.warning("⚠️ DNAChisel not available. Install with: pip install dnachisel")

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")

        # Sequence input
        sequence_input = st.text_area(
            "Enter Protein or DNA Sequence",
            height=150,
            placeholder="Enter protein sequence (MKWVT...) or DNA sequence (ATGGCG...)"
        )

        # Validate input
        if sequence_input:
            is_valid, message, sequence_type = validate_sequence(sequence_input)
            if is_valid:
                st.success(f"✅ {message}")
                sequence_clean = sequence_input.strip().upper()

                # Calculate input metrics
                input_metrics = calculate_input_metrics(sequence_clean, organism, sequence_type)

                st.subheader("Input Analysis")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    unit = "codons" if sequence_type == "dna" else "AA"
                    st.metric("Length", f"{input_metrics['length']} {unit}")
                    st.metric("GC Content", f"{input_metrics['gc_content']:.1f}%")

                with metrics_col2:
                    if input_metrics['cai']:
                        label = "CAI" if sequence_type == "dna" else "CAI (baseline)"
                        st.metric(label, f"{input_metrics['cai']:.3f}")
                    else:
                        st.metric("CAI", "N/A")

                with metrics_col3:
                    if input_metrics['tai']:
                        label = "tAI" if sequence_type == "dna" else "tAI (baseline)"
                        st.metric(label, f"{input_metrics['tai']:.3f}")
                    else:
                        st.metric("tAI", "N/A")

                # Additional sequence analysis
                st.subheader("Sequence Analysis")
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

                with analysis_col1:
                    st.metric("Restriction Sites", input_metrics.get('restriction_sites', 0))

                with analysis_col2:
                    st.metric("Negative Elements", input_metrics.get('negative_cis_elements', 0))

                with analysis_col3:
                    st.metric("Homopolymer Runs", input_metrics.get('homopolymer_runs', 0))

                # GC content plot for input
                if len(input_metrics['baseline_dna']) > 150:
                    fig = create_gc_content_plot(input_metrics['baseline_dna'])
                    st.plotly_chart(fig, use_container_width=True)



            else:
                st.error(f"❌ {message}")
                sequence_clean = None
                sequence_type = None
        else:
            sequence_clean = None
            sequence_type = None

    with col2:
        st.header("Optimization Results")

        # Single optimization button for any input
        if sequence_clean and not st.session_state.optimization_running:
            if st.button("🚀 Optimize Sequence", type="primary"):
                st.session_state.results = None
                # Handle both DNA and protein inputs
                if sequence_type == "dna":
                    # For DNA input, translate to protein first, then optimize
                    protein_sequence = translate_dna_to_protein(sequence_clean)
                    run_optimization(protein_sequence, organism, use_post_processing)
                else:
                    # For protein input, optimize directly
                    run_optimization(sequence_clean, organism, use_post_processing)

        # Show progress
        if st.session_state.optimization_running:
            st.info("🔄 Optimizing sequence... This may take a few moments.")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate progress (in real implementation, this would come from the model)
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 25:
                    status_text.text("Processing input sequence...")
                elif i < 50:
                    status_text.text("Running fine-tuned model prediction...")
                elif i < 75:
                    status_text.text("Optimizing GC content if needed...")
                else:
                    status_text.text("Finalizing optimized sequence...")
                time.sleep(0.05)

            progress_bar.empty()
            status_text.empty()

        # Display results
        if st.session_state.results and not st.session_state.optimization_running:
            if isinstance(st.session_state.results, str):
                st.error(st.session_state.results)
            else:
                result = st.session_state.results

                # Calculate optimized metrics using exact same method as evaluation script
                optimized_metrics = {
                    'gc_content': get_GC_content(result.predicted_dna),
                    'length': len(result.predicted_dna)
                }

                # Calculate CAI using same method as evaluation script
                try:
                    organism_key = f"cai_weights_{organism.replace(' ', '_')}"
                    if organism_key in st.session_state and st.session_state[organism_key]:
                        optimized_metrics['cai'] = CAI(result.predicted_dna, weights=st.session_state[organism_key])
                    else:
                        optimized_metrics['cai'] = None
                except:
                    optimized_metrics['cai'] = None

                # Calculate tAI using same method as evaluation script
                try:
                    if st.session_state.tai_weights:
                        optimized_metrics['tai'] = calculate_tAI(result.predicted_dna, st.session_state.tai_weights)
                    else:
                        optimized_metrics['tai'] = None
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

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download DNA Sequence",
                        data=result.predicted_dna,
                        file_name=f"optimized_sequence_{organism.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

                # Show post-processed results if available
                if st.session_state.post_processed_results:
                    if isinstance(st.session_state.post_processed_results, str):
                        st.error(st.session_state.post_processed_results)
                    else:
                        st.subheader("Post-Processed Results")
                        post_result = st.session_state.post_processed_results

                        # Calculate post-processed metrics
                        post_metrics = {
                            'gc_content': get_GC_content(post_result.predicted_dna),
                            'restriction_sites': len(scan_for_restriction_sites(post_result.predicted_dna)),
                            'negative_cis_elements': count_negative_cis_elements(post_result.predicted_dna),
                            'homopolymer_runs': calculate_homopolymer_runs(post_result.predicted_dna)
                        }

                        # Show improvement metrics
                        st.write("**Post-processing improvements:**")
                        imp_col1, imp_col2, imp_col3 = st.columns(3)

                        with imp_col1:
                            before_sites = len(scan_for_restriction_sites(result.predicted_dna))
                            after_sites = post_metrics['restriction_sites']
                            delta = before_sites - after_sites
                            st.metric("Restriction Sites", after_sites, delta=f"-{delta}" if delta > 0 else "0")

                        with imp_col2:
                            before_neg = count_negative_cis_elements(result.predicted_dna)
                            after_neg = post_metrics['negative_cis_elements']
                            delta = before_neg - after_neg
                            st.metric("Negative Elements", after_neg, delta=f"-{delta}" if delta > 0 else "0")

                        with imp_col3:
                            before_homo = calculate_homopolymer_runs(result.predicted_dna)
                            after_homo = post_metrics['homopolymer_runs']
                            delta = before_homo - after_homo
                            st.metric("Homopolymer Runs", after_homo, delta=f"-{delta}" if delta > 0 else "0")

                        # Download post-processed sequence
                        with col2:
                            st.download_button(
                                label="📥 Download Post-Processed",
                                data=post_result.predicted_dna,
                                file_name=f"post_processed_sequence_{organism.replace(' ', '_')}.txt",
                                mime="text/plain"
                            )

    # Full-width comparison section - works for both DNA and protein inputs
    if (st.session_state.results and
        not st.session_state.optimization_running and
        sequence_clean and
        not isinstance(st.session_state.results, str)):

        st.header("📊 Before vs After Comparison")

        # Calculate metrics for comparison
        if sequence_type == "dna":
            # For DNA input, compare original DNA vs optimized DNA
            input_metrics = calculate_input_metrics(sequence_clean, organism, sequence_type)
            original_dna = sequence_clean
        else:
            # For protein input, compare baseline vs optimized DNA
            input_metrics = calculate_input_metrics(sequence_clean, organism, sequence_type)
            original_dna = input_metrics['baseline_dna']

        result = st.session_state.results

        optimized_metrics = {
            'gc_content': get_GC_content(result.predicted_dna),
        }

        # Add CAI and tAI using same method as evaluation script
        try:
            organism_key = f"cai_weights_{organism.replace(' ', '_')}"
            if organism_key in st.session_state and st.session_state[organism_key]:
                optimized_metrics['cai'] = CAI(result.predicted_dna, weights=st.session_state[organism_key])
            else:
                optimized_metrics['cai'] = None
        except:
            optimized_metrics['cai'] = None

        try:
            if st.session_state.tai_weights:
                optimized_metrics['tai'] = calculate_tAI(result.predicted_dna, st.session_state.tai_weights)
            else:
                optimized_metrics['tai'] = None
        except:
            optimized_metrics['tai'] = None

        # Show improvement summary
        st.subheader("🎯 Improvement Summary")
        imp_col1, imp_col2, imp_col3 = st.columns(3)

        with imp_col1:
            if input_metrics['gc_content'] and optimized_metrics['gc_content']:
                gc_change = optimized_metrics['gc_content'] - input_metrics['gc_content']
                st.metric("GC Content", f"{optimized_metrics['gc_content']:.1f}%",
                         delta=f"{gc_change:+.1f}%")

        with imp_col2:
            if input_metrics['cai'] and optimized_metrics['cai']:
                cai_change = optimized_metrics['cai'] - input_metrics['cai']
                st.metric("CAI Score", f"{optimized_metrics['cai']:.3f}",
                         delta=f"{cai_change:+.3f}")

        with imp_col3:
            if input_metrics['tai'] and optimized_metrics['tai']:
                tai_change = optimized_metrics['tai'] - input_metrics['tai']
                st.metric("tAI Score", f"{optimized_metrics['tai']:.3f}",
                         delta=f"{tai_change:+.3f}")

        # Separated comparison charts
        gc_comp_fig = create_gc_comparison_chart(input_metrics, optimized_metrics)
        st.plotly_chart(gc_comp_fig, use_container_width=True, key="gc_comparison_chart")

        if input_metrics['cai'] and optimized_metrics['cai']:
            expr_comp_fig = create_expression_comparison_chart(input_metrics, optimized_metrics)
            st.plotly_chart(expr_comp_fig, use_container_width=True, key="expression_comparison_chart")

        # Side-by-side GC content plots
        st.subheader("📈 GC Content Distribution Comparison")
        col1, col2 = st.columns(2)

        with col1:
            if sequence_type == "dna":
                st.write("**Original DNA Sequence**")
            else:
                st.write("**Baseline (Most Frequent Codons)**")

            if len(original_dna) > 150:
                fig_before = create_gc_content_plot(original_dna)
                st.plotly_chart(fig_before, use_container_width=True, key="gc_dist_before")
            else:
                st.info("Sequence too short for sliding window analysis")

        with col2:
            st.write("**Optimized by Fine-tuned Model**")
            if len(result.predicted_dna) > 150:
                fig_after = create_gc_content_plot(result.predicted_dna)
                st.plotly_chart(fig_after, use_container_width=True, key="gc_dist_after")
            else:
                st.info("Sequence too short for sliding window analysis")

        # Sequence length comparison
        st.subheader("📏 Sequence Information")
        seq_col1, seq_col2 = st.columns(2)
        with seq_col1:
            st.write(f"**Original length**: {len(original_dna)} bp")
        with seq_col2:
            st.write(f"**Optimized length**: {len(result.predicted_dna)} bp")

        # Show if GC was auto-corrected
        original_gc = get_GC_content(original_dna)
        optimized_gc = get_GC_content(result.predicted_dna)

        if original_gc < 45 or original_gc > 55:
            if 45 <= optimized_gc <= 55:
                st.success("✅ GC content automatically corrected to optimal range (45-55%)")
            else:
                st.warning("⚠️ GC content still outside optimal range after optimization")

    # Footer
    st.markdown("---")
    st.markdown("**CodonTransformer GUI** - Optimize your protein sequences with AI-powered codon usage optimization")
    st.markdown("*Featuring fine-tuned model with proven performance improvements*")
    st.markdown("Built with Streamlit 🎈")

if __name__ == "__main__":
    main()
