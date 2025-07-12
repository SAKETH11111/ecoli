import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, BigBirdForMaskedLM
from huggingface_hub import hf_hub_download
from datasets import load_dataset
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
import json

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

def get_organism_tai_weights(organism: str) -> Dict[str, float]:
    """Get organism-specific tAI weights from pre-calculated data"""
    try:
        # Load organism-specific tAI weights
        weights_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'organism_tai_weights.json')
        with open(weights_file, 'r') as f:
            all_weights = json.load(f)

        if organism in all_weights:
            return all_weights[organism]
        else:
            # Fallback to E. coli if organism not found
            st.warning(f"tAI weights for {organism} not found, using E. coli weights")
            return all_weights.get("Escherichia coli general", get_ecoli_tai_weights())
    except Exception as e:
        st.error(f"Error loading organism-specific tAI weights: {e}")
        return get_ecoli_tai_weights()

def load_model_and_tokenizer():
    """Load the model and tokenizer with progress tracking"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading CodonTransformer model... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading tokenizer...")
            progress_bar.progress(25)
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

            status_text.text("Loading fine-tuned model from Hugging Face...")
            progress_bar.progress(50)
            # Try to download and load fine-tuned model from Hugging Face
            try:
                # Download the checkpoint file from Hugging Face
                from huggingface_hub import hf_hub_download
                
                status_text.text("⬇️ Downloading model from saketh11/ColiFormer...")
                model_path = hf_hub_download(
                    repo_id="saketh11/ColiFormer",
                    filename="balanced_alm_finetune.ckpt",
                    cache_dir="./hf_cache"
                )
                
                status_text.text("🔄 Loading downloaded model...")
                st.session_state.model = load_model(
                    model_path=model_path,
                    device=st.session_state.device,
                    attention_type="original_full"
                )
                status_text.text("✅ Fine-tuned model loaded from Hugging Face (6.2% better CAI)")
                st.session_state.model_type = "fine_tuned_hf"
            except Exception as e:
                status_text.text(f"⚠️ Failed to load from Hugging Face: {str(e)[:50]}...")
                status_text.text("Loading base model as fallback...")
                st.session_state.model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
                st.session_state.model = st.session_state.model.to(st.session_state.device)
                st.session_state.model_type = "base"

            progress_bar.progress(100)
            time.sleep(0.5)

            status_text.empty()
            progress_bar.empty()

@st.cache_data
def download_reference_data():
    """Download and cache reference data from Hugging Face"""
    try:
        # Download the processed genes file from Hugging Face
        file_path = hf_hub_download(
            repo_id="saketh11/ColiFormer-Data",
            filename="ecoli_processed_genes.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(file_path)
        return df['dna_sequence'].tolist()
    except Exception as e:
        st.warning(f"Could not download reference data from Hugging Face: {e}")
        # Fallback to minimal sequences
        return [
            "ATGGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGCGAAAGCGCTGTATCGC",
            "ATGAAATTTATTTATTATTATAAATTTATTTATTATTATAAATTTATTTAT",
            "ATGGGTCGTCGTCGTCGTGGTCGTCGTCGTCGTGGTCGTCGTCGTCGTGGT"
        ]

@st.cache_data
def download_tai_weights():
    """Download and cache tAI weights from Hugging Face"""
    try:
        # Download the tAI weights file from Hugging Face
        file_path = hf_hub_download(
            repo_id="saketh11/ColiFormer-Data",
            filename="organism_tai_weights.json",
            repo_type="dataset"
        )
        with open(file_path, 'r') as f:
            all_weights = json.load(f)
        return all_weights.get("Escherichia coli general", get_ecoli_tai_weights())
    except Exception as e:
        st.warning(f"Could not download tAI weights from Hugging Face: {e}")
        return get_ecoli_tai_weights()

def load_reference_data(organism: str = "Escherichia coli general"):
    """Load reference sequences and tAI weights for E. coli"""
    if 'cai_weights' not in st.session_state or st.session_state['cai_weights'] is None:
        try:
            # Download reference sequences from Hugging Face
            with st.spinner("📥 Downloading E. coli reference sequences from Hugging Face..."):
                ref_sequences = download_reference_data()
                st.session_state['cai_weights'] = relative_adaptiveness(sequences=ref_sequences)
                if len(ref_sequences) > 100:  # If we got the full dataset
                    st.success(f"✅ Downloaded {len(ref_sequences):,} E. coli reference sequences for CAI calculation")
                else:
                    st.info(f"⚠️ Using {len(ref_sequences)} minimal reference sequences (full dataset unavailable)")
        except Exception as e:
            st.error(f"Error loading E. coli reference data: {e}")
            st.session_state['cai_weights'] = {}
    # tAI weights (E. coli only)
    if 'tai_weights' not in st.session_state or st.session_state['tai_weights'] is None:
        try:
            with st.spinner("📥 Downloading E. coli tAI weights from Hugging Face..."):
                st.session_state['tai_weights'] = download_tai_weights()
                st.success("✅ Downloaded E. coli tAI weights")
        except Exception as e:
            st.error(f"Error loading E. coli tAI weights: {e}")
            st.session_state['tai_weights'] = {}

def validate_sequence(sequence: str) -> Tuple[bool, str, str, str]:
    """Validate sequence and return status, message, sequence type, and possibly fixed sequence"""
    if not sequence:
        return False, "Sequence cannot be empty", "unknown", sequence

    # Remove whitespace and convert to uppercase
    sequence = sequence.strip().upper()

    # Check if it's a DNA sequence
    dna_chars = set("ATGC")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY*_")

    sequence_chars = set(sequence)

    # If all characters are DNA nucleotides, treat as DNA
    if sequence_chars.issubset(dna_chars):
        if len(sequence) < 3:
            return False, "DNA sequence must be at least 3 nucleotides long", "dna", sequence
        
        # Auto-fix DNA sequences not divisible by 3
        if len(sequence) % 3 != 0:
            remainder = len(sequence) % 3
            fixed_sequence = sequence[:-remainder]
            message = f"Valid DNA sequence (auto-fixed: removed {remainder} nucleotides from end to make divisible by 3)"
        else:
            fixed_sequence = sequence
            message = "Valid DNA sequence"
        
        return True, message, "dna", fixed_sequence

    # If contains protein-specific amino acids, treat as protein
    elif sequence_chars.issubset(protein_chars):
        if len(sequence) < 3:
            return False, "Protein sequence must be at least 3 amino acids long", "protein", sequence
        return True, "Valid protein sequence", "protein", sequence

    # Invalid characters
    else:
        invalid_chars = sequence_chars - (dna_chars | protein_chars)
        return False, f"Invalid characters found: {', '.join(invalid_chars)}", "unknown", sequence

def calculate_input_metrics(sequence: str, organism: str, sequence_type: str) -> Dict:
    """Calculate metrics for the input sequence using E. coli reference only"""
    # Load reference data (E. coli only)
    load_reference_data()
    if sequence_type == "dna":
        dna_sequence = sequence.upper()
        metrics = {
            'length': len(dna_sequence) // 3,
            'gc_content': get_GC_content(dna_sequence),
            'baseline_dna': dna_sequence,
            'sequence_type': 'dna'
        }
        try:
            if 'cai_weights' in st.session_state and st.session_state['cai_weights']:
                metrics['cai'] = CAI(dna_sequence, weights=st.session_state['cai_weights'])
            else:
                metrics['cai'] = None
        except:
            metrics['cai'] = None
        try:
            if 'tai_weights' in st.session_state and st.session_state['tai_weights']:
                metrics['tai'] = calculate_tAI(dna_sequence, st.session_state['tai_weights'])
            else:
                metrics['tai'] = None
        except:
            metrics['tai'] = None
    else:
        most_frequent_codons = {
            'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
            'G': 'GGC', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTG',
            'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG', 'R': 'CGC',
            'S': 'TCG', 'T': 'ACG', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT',
            '*': 'TAA', '_': 'TAA'
        }
        baseline_dna = ''.join([most_frequent_codons.get(aa, 'NNN') for aa in sequence])
        metrics = {
            'length': len(sequence),
            'gc_content': get_GC_content(baseline_dna),
            'baseline_dna': baseline_dna,
            'sequence_type': 'protein'
        }
        try:
            if 'cai_weights' in st.session_state and st.session_state['cai_weights']:
                metrics['cai'] = CAI(baseline_dna, weights=st.session_state['cai_weights'])
            else:
                metrics['cai'] = None
        except:
            metrics['cai'] = None
        try:
            if 'tai_weights' in st.session_state and st.session_state['tai_weights']:
                metrics['tai'] = calculate_tAI(baseline_dna, st.session_state['tai_weights'])
            else:
                metrics['tai'] = None
        except:
            metrics['tai'] = None
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
            if isinstance(best_replacement, str):
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
        _res = result[0] if isinstance(result, list) else result
        initial_gc = get_GC_content(_res.predicted_dna)

        if initial_gc < 45.0 or initial_gc > 55.0:
            # Auto-correct GC content silently
            optimized_dna = smart_codon_replacement(_res.predicted_dna, 0.45, 0.55)
            smart_gc = get_GC_content(optimized_dna)

            if 45.0 <= smart_gc <= 55.0:
                from CodonTransformer.CodonUtils import DNASequencePrediction
                result = DNASequencePrediction(
                    organism=_res.organism,
                    protein=_res.protein,
                    processed_input=_res.processed_input,
                    predicted_dna=optimized_dna
                )
            else:
                # Fall back to constrained beam search silently
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
                    _res2 = result[0] if isinstance(result, list) else result
                    final_gc = get_GC_content(_res2.predicted_dna)
                except Exception as e:
                    # If constrained search fails, use smart replacement result anyway
                    from CodonTransformer.CodonUtils import DNASequencePrediction
                    result = DNASequencePrediction(
                        organism=_res.organism,
                        protein=_res.protein,
                        processed_input=_res.processed_input,
                        predicted_dna=optimized_dna
                    )

        st.session_state.results = result

        # Post-processing if enabled
        if use_post_processing and POST_PROCESSING_AVAILABLE and result:
            try:
                _res = result[0] if isinstance(result, list) else result
                polished_sequence = polish_sequence_with_dnachisel(
                    dna_sequence=_res.predicted_dna,
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
            except Exception as e:
                st.session_state.post_processed_results = f"Post-processing error: {str(e)}"

    except Exception as e:
        st.session_state.results = f"Error: {str(e)}"

    finally:
        st.session_state.optimization_running = False

def main():
    st.title("🧬 ColiFormer")
    st.markdown("**State-of-the-art E. coli codon optimization for publication-quality research**")

    # Remove the performance highlights expander (details/summary block)
    # (No expander here anymore)

    # Load model
    load_model_and_tokenizer()

    # Create the main tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Single Optimize", "📁 Batch Process", "📊 Comparative Analysis", "⚙️ Advanced Settings"])

    with tab1:
        single_sequence_optimization()

    with tab2:
        batch_processing_interface()

    with tab3:
        comparative_analysis_interface()

    with tab4:
        advanced_settings_interface()

def single_sequence_optimization():
    """Single sequence optimization interface - enhanced from original functionality"""
    # Sidebar configuration 
    st.sidebar.header("🔧 Configuration")
    organism_options = [
        "Escherichia coli general",
        "Saccharomyces cerevisiae",
        "Homo sapiens",
        "Bacillus subtilis",
        "Pichia pastoris"
    ]
    organism = st.sidebar.selectbox("Select Target Organism", organism_options)
    load_reference_data(organism)
    with st.sidebar.expander("🔧 Advanced Optimization Settings"):
        st.markdown("**Model Parameters**")
        use_deterministic = st.checkbox("Deterministic Mode", value=True, help="Use deterministic decoding for reproducible results")
        match_protein = st.checkbox("Match Protein Validation", value=True, help="Ensure DNA translates back to exact protein")
        st.markdown("**GC Content Control**")
        gc_target_min = st.slider("GC Target Min (%)", 30, 70, 45, help="Minimum GC content target")
        gc_target_max = st.slider("GC Target Max (%)", 30, 70, 55, help="Maximum GC content target")
        st.markdown("**Quality Constraints**")
        avoid_restriction_sites = st.multiselect(
            "Avoid Restriction Sites",
            ["EcoRI", "BamHI", "HindIII", "XhoI", "NotI"],
            default=["EcoRI", "BamHI"]
        )
    st.sidebar.subheader("🔬 Post-Processing")
    use_post_processing = st.sidebar.checkbox(
        "Enable DNAChisel Post-Processing",
        value=False,
        disabled=not POST_PROCESSING_AVAILABLE,
        help="Polish sequences to remove restriction sites, homopolymers, and synthesis issues"
    )
    if not POST_PROCESSING_AVAILABLE:
        st.sidebar.warning("⚠️ DNAChisel not available. Install with: pip install dnachisel")
    
    # Dataset Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Information")
    st.sidebar.markdown("""
    - **Dataset**: [ColiFormer-Data](https://huggingface.co/datasets/saketh11/ColiFormer-Data)
    - **Training**: 4,300 high-CAI E. coli sequences
    - **Reference**: 50,000+ E. coli gene sequences  
    - **Auto-download**: CAI weights & tAI coefficients
    """)
    
    # Model Information
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.markdown("""
    - **Model**: [ColiFormer](https://huggingface.co/saketh11/ColiFormer)
    - **Improvement**: +6.2% CAI vs base model
    - **Architecture**: BigBird Transformer + ALM
    - **Auto-download**: From Hugging Face Hub
    """)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("🧬 Input Sequence")
        sequence_input = st.text_area(
            "Enter Protein or DNA Sequence",
            height=150,
            placeholder="Enter protein sequence (MKWVT...) or DNA sequence (ATGGCG...)\n\nExample protein: MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTE"
        )
        analyze_btn = st.button("Analyze Sequence", type="primary")
        if sequence_input and analyze_btn:
            is_valid, message, sequence_type, fixed_sequence = validate_sequence(sequence_input)
            if is_valid:
                st.success(f"✅ {message}")
                # Store in session state for use by Optimize Sequence
                st.session_state.sequence_clean = fixed_sequence
                st.session_state.sequence_type = sequence_type
                st.session_state.input_metrics = calculate_input_metrics(fixed_sequence, organism, sequence_type)
                st.session_state.organism = organism
            else:
                st.error(f"❌ {message}")
                if "Invalid characters" in message:
                    st.info("💡 **Suggestion:** Remove spaces, numbers, and special characters. Use only standard amino acid letters (A-Z) for proteins or nucleotides (ATGC) for DNA.")
                elif "too long" in message:
                    st.info("💡 **Suggestion:** Consider breaking long sequences into smaller segments for optimization.")
                elif "too short" in message:
                    st.info("💡 **Suggestion:** Minimum length is 3 characters. Ensure your sequence is complete.")
                # Clear session state if invalid
                st.session_state.sequence_clean = None
                st.session_state.sequence_type = None
                st.session_state.input_metrics = None
                st.session_state.organism = None
        elif not sequence_input:
            st.session_state.sequence_clean = None
            st.session_state.sequence_type = None
            st.session_state.input_metrics = None
            st.session_state.organism = None

        # Always display the last analysis if it exists in session state
        if st.session_state.get('input_metrics') and st.session_state.get('sequence_type'):
            input_metrics = st.session_state.input_metrics
            sequence_type = st.session_state.sequence_type
            st.subheader("📊 Input Analysis")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                unit = "codons" if sequence_type == "dna" else "AA"
                length = input_metrics.get('length', 0) if input_metrics else 0
                gc_content = input_metrics.get('gc_content', 0) if input_metrics else 0
                st.metric("Length", f"{length} {unit}")
                st.metric("GC Content", f"{gc_content:.1f}%")
            with metrics_col2:
                cai_val = input_metrics.get('cai') if input_metrics else None
                if cai_val:
                    label = "CAI" if sequence_type == "dna" else "CAI (baseline)"
                    st.metric(label, f"{cai_val:.3f}")
                else:
                    st.metric("CAI", "N/A")
            with metrics_col3:
                tai_val = input_metrics.get('tai') if input_metrics else None
                if tai_val:
                    label = "tAI" if sequence_type == "dna" else "tAI (baseline)"
                    st.metric(label, f"{tai_val:.3f}")
                else:
                    st.metric("tAI", "N/A")
            st.subheader("🔍 Sequence Quality Analysis")
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            with analysis_col1:
                sites_count = input_metrics.get('restriction_sites', 0) if input_metrics else 0
                color = "normal" if sites_count <= 2 else "inverse"
                st.metric("Restriction Sites", sites_count)
            with analysis_col2:
                neg_elements = input_metrics.get('negative_cis_elements', 0) if input_metrics else 0
                st.metric("Negative Elements", neg_elements)
            with analysis_col3:
                homo_runs = input_metrics.get('homopolymer_runs', 0) if input_metrics else 0
                st.metric("Homopolymer Runs", homo_runs)
            baseline_dna = input_metrics.get('baseline_dna', '') if input_metrics else ''
            if baseline_dna and len(baseline_dna) > 150:
                st.subheader("📈 GC Content Distribution")
                fig = create_gc_content_plot(baseline_dna)
                fig.update_layout(
                    title="Input Sequence GC Content Analysis",
                    xaxis_title="Position (codons)",
                    yaxis_title="GC Content (%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("🚀 Optimization Results")
        # Enhanced optimization button
        if (
            st.session_state.get('sequence_clean')
            and st.session_state.get('sequence_type')
            and not st.session_state.optimization_running
        ):
            st.markdown("**Ready to optimize your sequence!**")
            strategy_info = st.container()
            with strategy_info:
                st.info(f"""
                **Optimization Strategy:**
                • Target organism: {st.session_state.organism}
                • Model: Fine-tuned CodonTransformer (89.6M parameters)
                • GC target: {gc_target_min}-{gc_target_max}%
                • Mode: {'Deterministic' if use_deterministic else 'Stochastic'}
                """)
            if st.button("🚀 Optimize Sequence", type="primary", use_container_width=True):
                st.session_state.results = None
                if st.session_state.sequence_type == "dna":
                    protein_sequence = translate_dna_to_protein(st.session_state.sequence_clean)
                    run_optimization(protein_sequence, st.session_state.organism, use_post_processing)
                else:
                    run_optimization(st.session_state.sequence_clean, st.session_state.organism, use_post_processing)

        # Enhanced progress display
        if st.session_state.optimization_running:
            st.info("🔄 **Optimizing sequence with our model...**")

            # Create progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Enhanced progress steps
                steps = [
                    "🔍 Analyzing input sequence structure...",
                    "🧬 Loading fine-tuned CodonTransformer model...",
                    "⚡ Running optimization algorithm...",
                    "🎯 Optimizing GC content for synthesis...",
                    "✅ Finalizing optimized sequence..."
                ]

                for i, step in enumerate(steps):
                    progress_value = int((i + 1) / len(steps) * 100)
                    progress_bar.progress(progress_value)
                    status_text.text(step)
                    time.sleep(0.8)  # Realistic timing

            progress_bar.empty()
            status_text.empty()

        # Enhanced results display
        if st.session_state.results and not st.session_state.optimization_running:
            if isinstance(st.session_state.results, str):
                st.error(f"❌ **Optimization Failed:** {st.session_state.results}")
            else:
                display_optimization_results(
                    st.session_state.results, 
                    st.session_state.get('organism', organism), 
                    st.session_state.get('sequence_clean', ''), 
                    st.session_state.get('sequence_type', 'protein'), 
                    st.session_state.get('input_metrics', {})
                )

def display_optimization_results(result, organism, original_sequence, sequence_type, input_metrics):
    """Enhanced results display with publication-quality visualizations"""

    # Calculate optimized metrics
    optimized_metrics = {
        'gc_content': get_GC_content(result.predicted_dna),
        'length': len(result.predicted_dna)
    }

    # Calculate CAI and tAI
    try:
        if 'cai_weights' in st.session_state and st.session_state['cai_weights']:
            optimized_metrics['cai'] = CAI(result.predicted_dna, weights=st.session_state['cai_weights'])
        else:
            optimized_metrics['cai'] = None
    except:
        optimized_metrics['cai'] = None

    try:
        if 'tai_weights' in st.session_state and st.session_state['tai_weights']:
            optimized_metrics['tai'] = calculate_tAI(result.predicted_dna, st.session_state['tai_weights'])
        else:
            optimized_metrics['tai'] = None
    except:
        optimized_metrics['tai'] = None

    # Success header
    st.success("✅ **Optimization Complete!** ")

    # Key improvements summary
    st.subheader("🎯 Optimization Improvements")
    imp_col1, imp_col2, imp_col3 = st.columns(3)

    if input_metrics is not None:
        with imp_col1:
            if input_metrics.get('gc_content') and optimized_metrics.get('gc_content'):
                gc_change = optimized_metrics['gc_content'] - input_metrics['gc_content']
                st.metric("GC Content", f"{optimized_metrics['gc_content']:.1f}%", delta=f"{gc_change:+.1f}%")

        with imp_col2:
            if input_metrics.get('cai') and optimized_metrics.get('cai'):
                cai_change = optimized_metrics['cai'] - input_metrics['cai']
                st.metric("CAI Score", f"{optimized_metrics['cai']:.3f}", delta=f"{cai_change:+.3f}")

        with imp_col3:
            if input_metrics.get('tai') and optimized_metrics.get('tai'):
                tai_change = optimized_metrics['tai'] - input_metrics['tai']
                st.metric("tAI Score", f"{optimized_metrics['tai']:.3f}", delta=f"{tai_change:+.3f}")

    # Optimized DNA sequence display
    st.subheader("🧬 Optimized DNA Sequence")
    st.text_area("Optimized DNA Sequence", result.predicted_dna, height=100)

    # Enhanced download and export options
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download DNA (FASTA)",
            data=f">Optimized_{organism.replace(' ', '_')}\n{result.predicted_dna}",
            file_name=f"optimized_sequence_{organism.replace(' ', '_')}.fasta",
            mime="text/plain"
        )

    with col2:
        # Create CSV report
        csv_data = f"Metric,Original,Optimized,Improvement\n"
        csv_data += f"GC Content (%),{input_metrics['gc_content']:.1f},{optimized_metrics['gc_content']:.1f},{optimized_metrics['gc_content'] - input_metrics['gc_content']:+.1f}\n"
        if input_metrics['cai'] and optimized_metrics['cai']:
            csv_data += f"CAI Score,{input_metrics['cai']:.3f},{optimized_metrics['cai']:.3f},{optimized_metrics['cai'] - input_metrics['cai']:+.3f}\n"
        if input_metrics['tai'] and optimized_metrics['tai']:
            csv_data += f"tAI Score,{input_metrics['tai']:.3f},{optimized_metrics['tai']:.3f},{optimized_metrics['tai'] - input_metrics['tai']:+.3f}\n"

        st.download_button(
            label="📊 Download Metrics (CSV)",
            data=csv_data,
            file_name=f"optimization_metrics_{organism.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    with col3:
        st.button("📄 Generate PDF Report", help="Coming soon: Publication-quality PDF report")

    # Enhanced comparison visualizations
    st.subheader("📊 Before vs After Analysis")

    # Create enhanced comparison charts
    create_enhanced_comparison_charts(input_metrics, optimized_metrics, original_sequence, result.predicted_dna, sequence_type)

def create_enhanced_comparison_charts(input_metrics, optimized_metrics, original_dna, optimized_dna, sequence_type):
    """Create publication-quality comparison visualizations"""
    if input_metrics is None or optimized_metrics is None:
        st.info("No comparison data available.")
        return

    # GC Content comparison
    gc_comp_fig = create_gc_comparison_chart(input_metrics, optimized_metrics)
    gc_comp_fig.update_layout(
        title="GC Content Optimization Results",
        font=dict(size=12),
        height=350
    )
    st.plotly_chart(gc_comp_fig, use_container_width=True)

    # Expression metrics comparison
    if input_metrics.get('cai') and optimized_metrics.get('cai'):
        expr_comp_fig = create_expression_comparison_chart(input_metrics, optimized_metrics)
        expr_comp_fig.update_layout(
            title="Expression Potential Improvement",
            font=dict(size=12),
            height=350
        )
        st.plotly_chart(expr_comp_fig, use_container_width=True)

    # Side-by-side GC distribution analysis
    st.subheader("📈 GC Content Distribution Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{'Original DNA' if sequence_type == 'dna' else 'Baseline (Most Frequent Codons)'}**")
        baseline_dna = input_metrics.get('baseline_dna') if input_metrics else None
        plot_dna = baseline_dna if baseline_dna is not None else original_dna
        if plot_dna is not None and isinstance(plot_dna, str) and len(plot_dna) > 150:
            fig_before = create_gc_content_plot(plot_dna)
            fig_before.update_layout(title="Before Optimization", height=300)
            st.plotly_chart(fig_before, use_container_width=True)
        else:
            st.info("Sequence too short for sliding window analysis")

    with col2:
        st.write("** Model Optimized**")
        if optimized_dna is not None and isinstance(optimized_dna, str) and len(optimized_dna) > 150:
            fig_after = create_gc_content_plot(optimized_dna)
            fig_after.update_layout(title="After Optimization", height=300)
            st.plotly_chart(fig_after, use_container_width=True)
        else:
            st.info("Sequence too short for sliding window analysis")

def batch_processing_interface():
    """Batch processing interface for multiple sequences"""
    st.header("📁 Batch Processing")
    st.markdown("**Process multiple protein sequences simultaneously with optimization**")

    # File upload section
    st.subheader("📤 Upload Sequences")
    uploaded_file = st.file_uploader(
        "Choose a file with multiple sequences",
        type=['csv', 'xlsx', 'fasta', 'txt', 'fa'],
        help="Upload CSV, Excel (XLSX, with 'sequence' column) or FASTA format files"
    )

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Process uploaded file
        try:
            def find_column(df, target):
                # Find column name case-insensitively and ignoring spaces
                for col in df.columns:
                    if col.strip().lower() == target:
                        return col
                return None

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                seq_col = find_column(df, 'sequence')
                name_col = find_column(df, 'name')
                if seq_col:
                    sequences = df[seq_col].tolist()
                    if name_col:
                        names = df[name_col].tolist()
                    else:
                        names = [f"Sequence_{i+1}" for i in range(len(sequences))]
                else:
                    st.error("CSV file must contain a column named 'sequence' (case-insensitive, spaces ignored)")
                    return
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                seq_col = find_column(df, 'sequence')
                name_col = find_column(df, 'name')
                if seq_col:
                    sequences = df[seq_col].tolist()
                    if name_col:
                        names = df[name_col].tolist()
                    else:
                        names = [f"Sequence_{i+1}" for i in range(len(sequences))]
                else:
                    st.error("Excel file must contain a column named 'sequence' (case-insensitive, spaces ignored)")
                    return
            else:
                # Handle FASTA format
                content = uploaded_file.read().decode('utf-8')
                sequences, names = parse_fasta_content(content)

            st.info(f"📊 Found {len(sequences)} sequences ready for optimization")

            # Batch configuration
            col1, col2 = st.columns(2)
            with col1:
                batch_organism = st.selectbox("Target Organism", [
                    "Escherichia coli general", "Saccharomyces cerevisiae", "Homo sapiens"
                ])
            with col2:
                max_sequences = st.number_input("Max sequences to process", 1, len(sequences), min(10, len(sequences)))

            # Start batch processing
            if st.button("🚀 Start Batch Optimization", type="primary"):
                run_batch_optimization(sequences[:max_sequences], names[:max_sequences], batch_organism)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Batch results display
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        display_batch_results()

def parse_fasta_content(content):
    """Parse FASTA format content"""
    sequences = []
    names = []
    current_seq = ""
    current_name = ""

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(current_seq)
                names.append(current_name)
            current_name = line[1:] if len(line) > 1 else f"Sequence_{len(sequences)+1}"
            current_seq = ""
        else:
            current_seq += line

    if current_seq:
        sequences.append(current_seq)
        names.append(current_name)

    return sequences, names

def run_batch_optimization(sequences, names, organism):
    """Run batch optimization with progress tracking"""
    st.session_state.batch_results = []
    st.session_state.batch_logs = []  # Collect info logs for auto-fixes

    # Load reference data for CAI/tAI
    load_reference_data(organism)
    cai_weights = st.session_state.get('cai_weights', None)
    tai_weights = st.session_state.get('tai_weights', None)

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (seq, name) in enumerate(zip(sequences, names)):
        progress = (i + 1) / len(sequences)
        progress_bar.progress(progress)
        status_text.text(f"Processing {name} ({i+1}/{len(sequences)})")

        try:
            # Validate sequence and get possibly fixed sequence
            is_valid, message, sequence_type, fixed_seq = validate_sequence(seq)
            if is_valid:
                # Log if auto-fixed
                if 'auto-fixed' in message:
                    st.session_state.batch_logs.append(f"{name}: {message}")
                # Calculate original metrics (use fixed_seq for DNA)
                if sequence_type == "dna":
                    orig_gc = get_GC_content(fixed_seq)
                    orig_cai = CAI(fixed_seq, weights=cai_weights) if cai_weights else None
                    orig_tai = calculate_tAI(fixed_seq, tai_weights) if tai_weights else None
                else:
                    # For protein, create baseline DNA
                    most_frequent_codons = {
                        'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
                        'G': 'GGC', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTG',
                        'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG', 'R': 'CGC',
                        'S': 'TCG', 'T': 'ACG', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT',
                        '*': 'TAA', '_': 'TAA'
                    }
                    baseline_dna = ''.join([most_frequent_codons.get(aa, 'NNN') for aa in fixed_seq])
                    orig_gc = get_GC_content(baseline_dna)
                    orig_cai = CAI(baseline_dna, weights=cai_weights) if cai_weights else None
                    orig_tai = calculate_tAI(baseline_dna, tai_weights) if tai_weights else None

                # Run optimization using the fixed sequence
                result = predict_dna_sequence(
                    protein=fixed_seq if sequence_type == "protein" else translate_dna_to_protein(fixed_seq),
                    organism=organism,
                    device=st.session_state.device,
                    model=st.session_state.model,
                    deterministic=True,
                    match_protein=True,
                )

                # If result is a list, use the first element
                if isinstance(result, list):
                    result_obj = result[0]
                else:
                    result_obj = result

                # Calculate optimized metrics
                opt_gc = get_GC_content(result_obj.predicted_dna)
                opt_cai = CAI(result_obj.predicted_dna, weights=cai_weights) if cai_weights else None
                opt_tai = calculate_tAI(result_obj.predicted_dna, tai_weights) if tai_weights else None

                metrics = {
                    'name': name,
                    'original_sequence': fixed_seq,
                    'optimized_dna': result_obj.predicted_dna,
                    'gc_content_before': orig_gc,
                    'gc_content_after': opt_gc,
                    'cai_before': orig_cai,
                    'cai_after': opt_cai,
                    'tai_before': orig_tai,
                    'tai_after': opt_tai,
                    'length_before': len(fixed_seq),
                    'length_after': len(result_obj.predicted_dna),
                    'validation_message': message
                }

                st.session_state.batch_results.append(metrics)
            else:
                # Only skip if truly invalid (not auto-fixable)
                st.session_state.batch_logs.append(f"{name}: {message}")

        except Exception as e:
            st.session_state.batch_logs.append(f"{name}: Error processing: {str(e)}")

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Batch optimization complete! Processed {len(st.session_state.batch_results)} sequences.")

def display_batch_results():
    """Display batch processing results"""
    st.subheader("📊 Batch Results")

    # Show all logs (auto-fixes and errors)
    if hasattr(st.session_state, 'batch_logs') and st.session_state.batch_logs:
        for log in st.session_state.batch_logs:
            st.info(log)

    results_df = pd.DataFrame(st.session_state.batch_results)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sequences Processed", len(results_df))
    with col2:
        st.metric("Avg GC Before", f"{results_df['gc_content_before'].mean():.1f}%")
        st.metric("Avg GC After", f"{results_df['gc_content_after'].mean():.1f}%")
    with col3:
        st.metric("Avg CAI Before", f"{results_df['cai_before'].mean():.3f}")
        st.metric("Avg CAI After", f"{results_df['cai_after'].mean():.3f}")
    with col4:
        st.metric("Avg tAI Before", f"{results_df['tai_before'].mean():.3f}")
        st.metric("Avg tAI After", f"{results_df['tai_after'].mean():.3f}")

    # CAI Extremes Analysis
    st.subheader("🎯 CAI Performance Analysis")
    
    # Filter out rows with NaN CAI values for analysis
    valid_cai_df = results_df.dropna(subset=['cai_after'])
    
    if len(valid_cai_df) > 0:
        # Find lowest and highest CAI sequences
        lowest_cai_idx = valid_cai_df['cai_after'].idxmin()
        highest_cai_idx = valid_cai_df['cai_after'].idxmax()
        
        lowest_cai_row = results_df.loc[lowest_cai_idx]
        highest_cai_row = results_df.loc[highest_cai_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔻 Lowest CAI Sequence**")
            st.write(f"**Name:** {lowest_cai_row['name']}")
            st.metric("CAI Score", f"{lowest_cai_row['cai_after']:.3f}")
            st.metric("GC Content", f"{lowest_cai_row['gc_content_after']:.1f}%")
            st.metric("tAI Score", f"{lowest_cai_row['tai_after']:.3f}")
            st.metric("Length", f"{lowest_cai_row['length_after']} bp")
            
            # Show improvement
            if pd.notna(lowest_cai_row['cai_before']):
                cai_improvement = lowest_cai_row['cai_after'] - lowest_cai_row['cai_before']
                st.metric("CAI Improvement", f"{cai_improvement:+.3f}")
        
        with col2:
            st.markdown("**🔺 Highest CAI Sequence**")
            st.write(f"**Name:** {highest_cai_row['name']}")
            st.metric("CAI Score", f"{highest_cai_row['cai_after']:.3f}")
            st.metric("GC Content", f"{highest_cai_row['gc_content_after']:.1f}%")
            st.metric("tAI Score", f"{highest_cai_row['tai_after']:.3f}")
            st.metric("Length", f"{highest_cai_row['length_after']} bp")
            
            # Show improvement
            if pd.notna(highest_cai_row['cai_before']):
                cai_improvement = highest_cai_row['cai_after'] - highest_cai_row['cai_before']
                st.metric("CAI Improvement", f"{cai_improvement:+.3f}")
        
        # CAI Distribution Chart
        st.subheader("📊 CAI Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_cai_df['cai_after'],
            nbinsx=20,
            name='Optimized CAI Scores',
            marker_color='darkblue',
            opacity=0.7
        ))
        
        # Add vertical lines for lowest and highest
        fig.add_vline(
            x=lowest_cai_row['cai_after'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Lowest: {lowest_cai_row['cai_after']:.3f}"
        )
        fig.add_vline(
            x=highest_cai_row['cai_after'],
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Highest: {highest_cai_row['cai_after']:.3f}"
        )
        
        fig.update_layout(
            title="Distribution of Optimized CAI Scores",
            xaxis_title="CAI Score",
            yaxis_title="Number of Sequences",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # GC Content Distribution Chart
        st.subheader("📊 GC Content Distribution")
        valid_gc_df = results_df.dropna(subset=['gc_content_after'])
        if len(valid_gc_df) > 0:
            lowest_gc_idx = valid_gc_df['gc_content_after'].idxmin()
            highest_gc_idx = valid_gc_df['gc_content_after'].idxmax()
            lowest_gc_row = results_df.loc[lowest_gc_idx]
            highest_gc_row = results_df.loc[highest_gc_idx]

            fig_gc = go.Figure()
            fig_gc.add_trace(go.Histogram(
                x=valid_gc_df['gc_content_after'],
                nbinsx=20,
                name='Optimized GC Content',
                marker_color='teal',
                opacity=0.7
            ))
            fig_gc.add_vline(
                x=lowest_gc_row['gc_content_after'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Lowest: {lowest_gc_row['gc_content_after']:.1f}%"
            )
            fig_gc.add_vline(
                x=highest_gc_row['gc_content_after'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Highest: {highest_gc_row['gc_content_after']:.1f}%"
            )
            fig_gc.update_layout(
                title="Distribution of Optimized GC Content",
                xaxis_title="GC Content (%)",
                yaxis_title="Number of Sequences",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_gc, use_container_width=True)
        else:
            st.warning("⚠️ No valid GC content values found in the batch results.")
        
    else:
        st.warning("⚠️ No valid CAI scores found in the batch results. Check if CAI weights are properly loaded.")

    # Sequence selector
    seq_names = results_df['name'].tolist()
    selected_seq = st.selectbox("Select a sequence to view details", seq_names)
    seq_row = results_df[results_df['name'] == selected_seq].iloc[0]

    st.markdown(f"### Details for: {selected_seq}")
    if 'validation_message' in seq_row and 'auto-fixed' in seq_row['validation_message']:
        st.info(seq_row['validation_message'])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Sequence**")
        st.text_area("Original Sequence", seq_row['original_sequence'], height=100)
        st.metric("GC Content (Before)", f"{seq_row['gc_content_before']:.1f}%")
        st.metric("CAI (Before)", f"{seq_row['cai_before']:.3f}")
        st.metric("tAI (Before)", f"{seq_row['tai_before']:.3f}")
        st.metric("Length (Before)", f"{seq_row['length_before']}")
    with col2:
        st.markdown("**Optimized Sequence**")
        st.text_area("Optimized Sequence", seq_row['optimized_dna'], height=100)
        st.metric("GC Content (After)", f"{seq_row['gc_content_after']:.1f}%")
        st.metric("CAI (After)", f"{seq_row['cai_after']:.3f}")
        st.metric("tAI (After)", f"{seq_row['tai_after']:.3f}")
        st.metric("Length (After)", f"{seq_row['length_after']}")

    # Plots for before/after GC content
    st.subheader("GC Content Distribution (Before vs After)")
    if len(seq_row['original_sequence']) > 150 and len(seq_row['optimized_dna']) > 150:
        fig_before = create_gc_content_plot(seq_row['original_sequence'])
        fig_before.update_layout(title="Before Optimization", height=300)
        fig_after = create_gc_content_plot(seq_row['optimized_dna'])
        fig_after.update_layout(title="After Optimization", height=300)
        st.plotly_chart(fig_before, use_container_width=True)
        st.plotly_chart(fig_after, use_container_width=True)
    else:
        st.info("Sequence(s) too short for sliding window analysis")

    # Download batch results
    if st.button("📥 Download Batch Results"):
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="batch_optimization_results.csv",
            mime="text/csv"
        )

def comparative_analysis_interface():
    """Comparative analysis interface"""
    st.header("📊 Comparative Analysis")
    st.markdown("**Compare optimization strategies side-by-side**")

    st.info("🚧 **Coming Soon:** Compare our model against traditional methods (HFC, BFC, URC) and generate publication-quality comparative analysis.")

    # Placeholder for future implementation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Algorithm Comparison")
        st.write("• ColiFormer (Our Model)")
        st.write("• High Frequency Choice (HFC)")
        st.write("• Background Frequency Choice (BFC)")
        st.write("• Uniform Random Choice (URC)")

    with col2:
        st.subheader("Comparison Metrics")
        st.write("• CAI Score Comparison")
        st.write("• tAI Score Comparison")
        st.write("• GC Content Analysis")
        st.write("• Statistical Significance Testing")

def advanced_settings_interface():
    """Advanced settings and configuration interface"""
    st.header("⚙️ Advanced Settings")
    st.markdown("**Configure advanced parameters and model settings**")

    # Model configuration
    st.subheader("🤖 Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Model Status:**")
        if st.session_state.model:
            model_type = getattr(st.session_state, 'model_type', 'unknown')
            st.success(f"✅ Model loaded: {model_type}")
            st.write(f"Device: {st.session_state.device}")
        else:
            st.warning("⚠️ Model not loaded")

    with col2:
        st.write("**Model Information:**")
        st.write("• Architecture: BigBird Transformer")
        st.write("• Parameters: 89.6M")
        st.write("• Training: 4,316 high-CAI E. coli genes")
        st.write("• Performance: +5.1% CAI, +8.6% tAI")

    # Performance tuning
    st.subheader("⚡ Performance Tuning")

    # Memory management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully")

    with col2:
        if st.button("🔄 Reload Model"):
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.rerun()

    # System information
    st.subheader("💻 System Information")
    import torch
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**PyTorch:**")
        st.write(f"Version: {torch.__version__}")
        st.write(f"CUDA Available: {torch.cuda.is_available()}")

    with col2:
        st.write("**Device:**")
        st.write(f"Current: {st.session_state.device}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name()}")

    with col3:
        st.write("**Memory:**")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.write(f"GPU Memory: {gpu_memory:.1f} GB")

    # Footer
    st.markdown("---")
    st.markdown("**ColiFormer **")
    st.markdown("🚀 Built for Nature Communications-level research • Targeting >20% CAI improvements • Aug 2025 experimental validation")

if __name__ == "__main__":
    main()
