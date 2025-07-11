"""
File: CodonPostProcessing.py
---------------------------
Post-processing utilities for codon optimization using DNAChisel.
This module provides sequence polishing capabilities to fix restriction sites,
homopolymers, and other constraints while preserving CAI and GC content.
"""

import warnings
import numpy as np

try:
    from dnachisel import (
        DnaOptimizationProblem,
        AvoidPattern,
        EnforceGCContent,
        EnforceTranslation,
        CodonOptimize,
    )
    DNACHISEL_AVAILABLE = True
except ImportError:
    DNACHISEL_AVAILABLE = False
    # This warning will be shown when the module is first imported.
    warnings.warn(
        "DNAChisel is not installed. Post-processing features will be disabled."
    )

def polish_sequence_with_dnachisel(
    dna_sequence: str,
    protein_sequence: str,
    gc_bounds: tuple = (45.0, 55.0),
    cai_species: str = "e_coli",
    avoid_homopolymers_length: int = 6,
    enzymes_to_avoid: list = None
):
    """
    Polishes a DNA sequence using DNAChisel to meet lab synthesis constraints.
    """
    if not DNACHISEL_AVAILABLE:
        warnings.warn("DNAChisel not available, skipping post-processing.")
        return dna_sequence

    if enzymes_to_avoid is None:
        # Common cloning enzymes
        enzymes_to_avoid = ["EcoRI", "XbaI", "SpeI", "PstI", "NotI"]

    try:
        # Start with the basic, essential constraints
        constraints = [
            EnforceTranslation(translation=protein_sequence),
            EnforceGCContent(mini=gc_bounds[0] / 100.0, maxi=gc_bounds[1] / 100.0),
        ]

        # Add enzyme avoidance constraints safely
        for enzyme in enzymes_to_avoid:
            try:
                # This is the modern way to avoid enzyme sites
                constraints.append(AvoidPattern.from_enzyme_name(enzyme))
            except Exception:
                warnings.warn(f"Could not find enzyme '{enzyme}' in DNAChisel library.")

        # Add homopolymer avoidance constraints
        for base in "ATGC":
            constraints.append(AvoidPattern(base * avoid_homopolymers_length))

        # Define the optimization problem
        problem = DnaOptimizationProblem(
            sequence=dna_sequence,
            constraints=constraints,
            objectives=[CodonOptimize(species=cai_species, method="match_codon_usage")]
        )

        # Solve the problem
        problem.resolve_constraints()
        problem.optimize()

        # Return the polished sequence
        return problem.sequence

    except Exception as e:
        warnings.warn(f"DNAChisel post-processing failed with an error: {e}")
        # Return the original sequence if polishing fails
        return dna_sequence
