"""
File: CodonPostProcessing.py
---------------------------
Post-processing utilities for codon optimization using DNAChisel.
This module provides sequence polishing capabilities to fix restriction sites,
homopolymers, and other constraints while preserving CAI and GC content.
"""

import warnings
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    from dnachisel import (
        DnaOptimizationProblem,
        AvoidPattern,
        AvoidRestrictionSites,
        EnforceGCContent,
        MaximizeCAI,
        AvoidHairpins,
        AvoidHardToSynthesize,
        AvoidRareCodons,
        AvoidRepeats,
        CodonOptimize,
        EnforceTranslation,
        reverse_translate,
    )
    from dnachisel.builtin_specifications import MaximizeCAI as MaxCAI_spec
    DNACHISEL_AVAILABLE = True
    print('DNAChisel imported successfully!')
except ImportError as e:
    DNACHISEL_AVAILABLE = False
    print('DNAChisel import failed:', str(e))
    warnings.warn(
        "DNAChisel not available. Install with: pip install dnachisel. "
        "Post-processing features will be disabled."
    )

from CodonTransformer.CodonEvaluation import (
    get_GC_content,
    scan_for_restriction_sites,
    count_negative_cis_elements,
    calculate_homopolymer_runs,
)


def polish_sequence_with_dnachisel(
    dna_sequence: str,
    protein_sequence: str,
    gc_bounds: Tuple[float, float] = (50.0, 54.0),
    avoid_restriction_sites: List[str] = None,
    avoid_patterns: List[str] = None,
    maximize_cai: bool = True,
    cai_species: str = "e_coli",
    avoid_homopolymers: int = 6,
    max_iterations: int = 100,
    temperature: float = 4.0,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Polish a DNA sequence using DNAChisel to fix restriction sites, homopolymers,
    and other constraints while preserving translation and optimizing CAI.
    
    Args:
        dna_sequence (str): The input DNA sequence to polish
        protein_sequence (str): The target protein sequence for translation validation
        gc_bounds (Tuple[float, float]): GC content bounds (min%, max%)
        avoid_restriction_sites (List[str]): Restriction sites to avoid
        avoid_patterns (List[str]): Additional patterns to avoid
        maximize_cai (bool): Whether to maximize CAI during optimization
        cai_species (str): Species for CAI calculation
        avoid_homopolymers (int): Maximum homopolymer length to allow
        max_iterations (int): Maximum optimization iterations
        temperature (float): Optimization temperature
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[str, Dict[str, Any]]: (polished_sequence, optimization_report)
    """
    print('Starting DNAChisel polish for sequence of length:', len(dna_sequence))
    if not DNACHISEL_AVAILABLE:
        print('DNAChisel not available - skipping')
        return dna_sequence, {"error": "DNAChisel not available"}
    
    # Default restriction sites to avoid (common cloning sites)
    if avoid_restriction_sites is None:
        avoid_restriction_sites = [
            "GAATTC",  # EcoRI
            "GGATCC",  # BamHI
            "AAGCTT",  # HindIII
            "GTCGAC",  # SalI
            "CTGCAG",  # PstI
            "TCGCGA",  # NruI
            "GCGGCCGC",  # NotI
            "TTAATTAA",  # PacI
        ]
    
    # Default patterns to avoid
    if avoid_patterns is None:
        avoid_patterns = [
            "TATAAT",  # -10 box (Pribnow box)
            "TTGACA",  # -35 box
            "AGCTAGT",  # Negative cis-regulatory element
            "AAAAA",   # Poly-A run
            "TTTTT",   # Poly-T run
            "GGGGG",   # Poly-G run
            "CCCCC",   # Poly-C run
        ]
    
    try:
        # Create optimization problem
        problem = DnaOptimizationProblem(
            sequence=dna_sequence,
            constraints=[
                # Ensure the sequence translates to the target protein
                EnforceTranslation(translation=protein_sequence),
                
                # Enforce GC content bounds
                EnforceGCContent(mini=gc_bounds[0]/100, maxi=gc_bounds[1]/100),
                
                # Avoid restriction sites
                AvoidRestrictionSites(enzyme_names=[
                    site for site in avoid_restriction_sites
                ]),
                
                # Avoid problematic patterns
                *[AvoidPattern(pattern=pattern) for pattern in avoid_patterns],
                
                # Avoid homopolymer runs
                AvoidRepeats(n=avoid_homopolymers),
                
                # Avoid hairpins and secondary structures
                AvoidHairpins(),
                
                # Avoid hard-to-synthesize sequences
                AvoidHardToSynthesize(),
            ],
            objectives=[
                # Maximize CAI if requested
                MaximizeCAI(species=cai_species) if maximize_cai else None,
                
                # Minimize rare codons
                AvoidRareCodons(species=cai_species),
            ],
            logger=None,  # Suppress optimization logs
        )
        
        # Remove None objectives
        problem.objectives = [obj for obj in problem.objectives if obj is not None]
        
        # Set optimization parameters
        if seed is not None:
            np.random.seed(seed)
        
        # Solve the optimization problem
        problem.resolve_constraints_locally(
            max_iterations=max_iterations,
            temperature=temperature,
        )
        
        # Get the optimized sequence
        optimized_sequence = str(problem.sequence)
        
        # Create optimization report
        report = {
            "status": "success",
            "original_length": len(dna_sequence),
            "optimized_length": len(optimized_sequence),
            "constraints_satisfied": problem.all_constraints_pass(),
            "objective_scores": {},
            "iterations_used": max_iterations,  # DNAChisel doesn't return actual iterations
        }
        
        # Calculate objective scores
        for obj in problem.objectives:
            try:
                score = obj.evaluate(problem)
                report["objective_scores"][obj.__class__.__name__] = score
            except Exception as e:
                report["objective_scores"][obj.__class__.__name__] = f"Error: {str(e)}"
        
        # Validate translation
        if translate_dna_to_protein(optimized_sequence) != protein_sequence:
            warnings.warn(
                "Optimized sequence does not translate to target protein. "
                "Returning original sequence."
            )
            return dna_sequence, {
                "status": "failed",
                "reason": "Translation validation failed"
            }
        
        print('Optimization successful! Constraints satisfied:', problem.all_constraints_pass())
        return optimized_sequence, report
        
    except Exception as e:
        print('Optimization failed with error:', str(e))
        return dna_sequence, {"error": str(e)}


def translate_dna_to_protein(dna_sequence: str) -> str:
    """
    Translate DNA sequence to protein sequence using the standard genetic code.
    """
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


def apply_pareto_filtering(
    sequences: List[str],
    metrics: List[Dict[str, float]],
    objective_directions: Dict[str, str] = None,
) -> Tuple[List[str], List[Dict[str, float]], List[int]]:
    """
    Apply Pareto frontier filtering to select non-dominated sequences.
    
    Args:
        sequences (List[str]): List of DNA sequences
        metrics (List[Dict[str, float]]): List of metric dictionaries for each sequence
        objective_directions (Dict[str, str]): Direction for each objective ('max' or 'min')
        
    Returns:
        Tuple[List[str], List[Dict[str, float]], List[int]]: 
            (filtered_sequences, filtered_metrics, pareto_indices)
    """
    if not sequences or not metrics:
        return [], [], []
    
    # Default objective directions for codon optimization
    if objective_directions is None:
        objective_directions = {
            'cai': 'max',
            'tai': 'max',
            'gc_content': 'target',  # Special case - we want to be close to target
            'restriction_sites': 'min',
            'neg_cis_elements': 'min',
            'homopolymer_runs': 'min',
            'dtw_distance': 'min',
            'enc': 'min',
            'cpb': 'min',
            'scuo': 'min',
        }
    
    # Convert metrics to numpy array
    metric_names = list(metrics[0].keys())
    metric_matrix = np.array([
        [m[name] for name in metric_names] 
        for m in metrics
    ])
    
    # Apply Pareto filtering
    try:
        from paretoset import paretoset
        
        # Convert directions to paretoset format
        sense = []
        for name in metric_names:
            direction = objective_directions.get(name, 'max')
            if direction == 'max':
                sense.append('max')
            elif direction == 'min':
                sense.append('min')
            else:  # target - treat as minimization of absolute deviation
                sense.append('min')
        
        # Get Pareto frontier
        pareto_mask = paretoset(metric_matrix, sense=sense)
        pareto_indices = np.where(pareto_mask)[0].tolist()
        
        # Filter sequences and metrics
        filtered_sequences = [sequences[i] for i in pareto_indices]
        filtered_metrics = [metrics[i] for i in pareto_indices]
        
        return filtered_sequences, filtered_metrics, pareto_indices
        
    except ImportError:
        warnings.warn(
            "paretoset not available. Install with: pip install paretoset. "
            "Returning all sequences."
        )
        return sequences, metrics, list(range(len(sequences)))
    except Exception as e:
        warnings.warn(f"Pareto filtering failed: {str(e)}. Returning all sequences.")
        return sequences, metrics, list(range(len(sequences)))


def calculate_sequence_metrics(
    dna_sequence: str,
    protein_sequence: str,
    cai_weights: Dict[str, float],
    tai_weights: Dict[str, float],
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    reference_profile: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a DNA sequence.
    
    Args:
        dna_sequence (str): DNA sequence to evaluate
        protein_sequence (str): Target protein sequence
        cai_weights (Dict[str, float]): CAI weights for the organism
        tai_weights (Dict[str, float]): tAI weights for the organism
        codon_frequencies (Dict): Codon frequencies for the organism
        reference_profile (Optional[List[float]]): Reference profile for DTW calculation
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    from CAI import CAI
    from CodonTransformer.CodonEvaluation import (
        calculate_tAI,
        get_min_max_profile,
        calculate_dtw_distance,
        calculate_ENC,
        calculate_CPB,
        calculate_SCUO,
    )
    
    metrics = {}
    
    # Primary expression metrics
    metrics['cai'] = CAI(dna_sequence, weights=cai_weights)
    metrics['tai'] = calculate_tAI(dna_sequence, tai_weights)
    metrics['gc_content'] = get_GC_content(dna_sequence)
    
    # Health metrics
    metrics['restriction_sites'] = scan_for_restriction_sites(dna_sequence)
    metrics['neg_cis_elements'] = count_negative_cis_elements(dna_sequence)
    metrics['homopolymer_runs'] = calculate_homopolymer_runs(dna_sequence)
    
    # Advanced metrics
    try:
        min_max_profile = get_min_max_profile(dna_sequence, codon_frequencies)
        if reference_profile and min_max_profile:
            metrics['dtw_distance'] = calculate_dtw_distance(min_max_profile, reference_profile)
        else:
            metrics['dtw_distance'] = 0.0
    except Exception:
        metrics['dtw_distance'] = 0.0
    
    # Enhanced codon usage metrics
    try:
        metrics['enc'] = calculate_ENC(dna_sequence)
    except Exception:
        metrics['enc'] = 45.0  # Default E. coli value
    
    try:
        metrics['cpb'] = calculate_CPB(dna_sequence)
    except Exception:
        metrics['cpb'] = 0.0
    
    try:
        metrics['scuo'] = calculate_SCUO(dna_sequence)
    except Exception:
        metrics['scuo'] = 0.5
    
    return metrics


def enhanced_sequence_generation(
    protein_sequence: str,
    model,
    tokenizer,
    device,
    organism: str = "Escherichia coli general",
    beam_size: int = 20,
    gc_bounds: Tuple[float, float] = (50.0, 54.0),
    use_dnachisel_polish: bool = True,
    use_pareto_filtering: bool = True,
    cai_weights: Optional[Dict[str, float]] = None,
    tai_weights: Optional[Dict[str, float]] = None,
    codon_frequencies: Optional[Dict] = None,
    reference_profile: Optional[List[float]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate enhanced codon-optimized sequences using multiple improvement techniques.
    
    Args:
        protein_sequence (str): Input protein sequence
        model: Trained CodonTransformer model
        tokenizer: Model tokenizer
        device: PyTorch device
        organism (str): Target organism
        beam_size (int): Size of beam search
        gc_bounds (Tuple[float, float]): GC content bounds
        use_dnachisel_polish (bool): Whether to apply DNAChisel polishing
        use_pareto_filtering (bool): Whether to apply Pareto filtering
        cai_weights (Optional[Dict[str, float]]): CAI weights
        tai_weights (Optional[Dict[str, float]]): tAI weights
        codon_frequencies (Optional[Dict]): Codon frequencies
        reference_profile (Optional[List[float]]): Reference profile for DTW
        
    Returns:
        Tuple[str, Dict[str, Any]]: (best_sequence, generation_report)
    """
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    
    # Generate multiple candidates using beam search
    try:
        candidates = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=device,
            model=model,
            tokenizer=tokenizer,
            use_constrained_search=True,
            gc_bounds=(gc_bounds[0]/100, gc_bounds[1]/100),
            beam_size=beam_size,
            deterministic=True,
            match_protein=True,
        )
        
        # Handle single candidate case
        if not isinstance(candidates, list):
            candidates = [candidates]
            
        # Extract DNA sequences
        candidate_sequences = [c.predicted_dna for c in candidates]
        
    except Exception as e:
        # Fallback to single sequence generation
        warnings.warn(f"Beam search failed: {str(e)}. Using single sequence generation.")
        candidate = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=device,
            model=model,
            tokenizer=tokenizer,
            deterministic=True,
            match_protein=True,
        )
        candidate_sequences = [candidate.predicted_dna]
    
    # Apply DNAChisel polishing if requested
    if use_dnachisel_polish:
        polished_sequences = []
        polish_reports = []
        
        for seq in candidate_sequences:
            polished_seq, polish_report = polish_sequence_with_dnachisel(
                dna_sequence=seq,
                protein_sequence=protein_sequence,
                gc_bounds=gc_bounds,
                maximize_cai=True,
                seed=42,
            )
            polished_sequences.append(polished_seq)
            polish_reports.append(polish_report)
        
        candidate_sequences = polished_sequences
    
    # Calculate metrics for all candidates
    if cai_weights and tai_weights and codon_frequencies:
        candidate_metrics = []
        for seq in candidate_sequences:
            metrics = calculate_sequence_metrics(
                dna_sequence=seq,
                protein_sequence=protein_sequence,
                cai_weights=cai_weights,
                tai_weights=tai_weights,
                codon_frequencies=codon_frequencies,
                reference_profile=reference_profile,
            )
            candidate_metrics.append(metrics)
        
        # Apply Pareto filtering if requested
        if use_pareto_filtering:
            filtered_sequences, filtered_metrics, pareto_indices = apply_pareto_filtering(
                sequences=candidate_sequences,
                metrics=candidate_metrics,
            )
            
            if filtered_sequences:
                candidate_sequences = filtered_sequences
                candidate_metrics = filtered_metrics
        
        # Select best sequence based on weighted score
        best_idx = 0
        best_score = -float('inf')
        
        for i, metrics in enumerate(candidate_metrics):
            # Weighted scoring (adjust weights as needed)
            score = (
                metrics['cai'] * 0.3 +
                metrics['tai'] * 0.3 +
                (1 - abs(metrics['gc_content'] - 52) / 52) * 0.2 +  # Closer to 52% is better
                (1 - metrics['restriction_sites'] / 10) * 0.1 +  # Fewer is better
                (1 - metrics['neg_cis_elements'] / 10) * 0.1   # Fewer is better
            )
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        best_sequence = candidate_sequences[best_idx]
        best_metrics = candidate_metrics[best_idx]
        
    else:
        # Fallback to first sequence if no metrics available
        best_sequence = candidate_sequences[0]
        best_metrics = {}
    
    # Create generation report
    report = {
        'status': 'success',
        'candidates_generated': len(candidate_sequences),
        'dnachisel_applied': use_dnachisel_polish,
        'pareto_filtering_applied': use_pareto_filtering,
        'best_metrics': best_metrics,
        'beam_size_used': beam_size,
    }
    
    return best_sequence, report