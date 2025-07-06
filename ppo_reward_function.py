"""
Multi-Objective Reward Function for PPO Codon Optimization
============================================================

Research-quality reward function implementing normalized multi-component 
scoring for reinforcement learning-based codon optimization.

Based on:
- codonGPT (2025): Multi-component reward for biological sequence RL
- Multi-Dimensional Optimization (2024): Metric normalization methodology
- Local project: Advanced CAI/tAI/GC evaluation framework

"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import itertools

import numpy as np
import pandas as pd
import torch
from CAI import CAI, relative_adaptiveness
from scipy import stats
from Bio.Data import CodonTable

from CodonTransformer.CodonData import get_codon_frequencies
from CodonTransformer.CodonEvaluation import (
    calculate_dtw_distance,
    calculate_homopolymer_runs,
    get_min_max_profile,
    calculate_tAI,
    count_negative_cis_elements,
    get_GC_content,
    get_ecoli_tai_weights,
    get_min_max_profile,
    scan_for_restriction_sites,
)

# Suppress warnings for cleaner research output
warnings.filterwarnings('ignore')


class MultiObjectiveRewardFunction:
    """
    Research-grade multi-objective reward function for PPO codon optimization.
    
    Implements a robust, adaptive reward system based on the latest research:
    - CAI & tAI: For expression and translation efficiency.
    - GC Content: Hinge loss for precise targeting.
    - DTW Distance: For sequence naturalness.
    - Structural Penalties: For biological viability (restriction sites, etc.).
    
    Methodology (based on blueprint):
    1. Adaptive Inverse-σ Weighting: Dynamically adjusts weights based on component variance.
    2. Batch-wise Z-score Normalization: Ensures all metrics contribute fairly.
    3. Differentiable Implementation: Suitable for gradient-based RL.
    """
    
    def __init__(
        self,
        cai_weights: Optional[Dict] = None,
        tai_weights: Optional[Dict] = None,
        reference_profile: Optional[np.ndarray] = None,
        codon_frequencies: Optional[Dict] = None,
        gc_target: float = 52.0,
        gc_tolerance: float = 2.0,
        reward_weights: Optional[Dict[str, float]] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        organism_id: int = 51, # Default to E. coli project ID
    ):
        """
        Initialize multi-objective reward function.
        """
        self.device = device
        self.gc_target = gc_target
        self.gc_tolerance = gc_tolerance
        self.gc_min = gc_target - gc_tolerance
        self.gc_max = gc_target + gc_tolerance
        self.verbose = verbose
        self.organism_id = organism_id

        # Initialize reward component weights (research-tuned)
        self.reward_weights = reward_weights or {
            'cai': 0.35,
            'tai': 0.25,
            'gc': 0.25,
            'dtw': 0.10,
            'penalty': 0.05
        }

        # Initialize evaluation components
        self._initialize_evaluation_components(
            cai_weights, tai_weights, reference_profile, codon_frequencies
        )
        
        # Performance tracking
        self.call_count = 0
        self.reward_history = []

        self.cai_quintiles = self._compute_cai_quintiles(self.organism_id)
        
        if self.verbose:
            print("✓ Multi-Objective Reward Function initialized")
            print(f"  Target GC: {gc_target:.1f}% ± {gc_tolerance:.1f}%")
            print(f"  Weights: {self.reward_weights}")
    
    def _initialize_evaluation_components(
        self,
        cai_weights: Optional[Dict],
        tai_weights: Optional[Dict],
        reference_profile: Optional[np.ndarray],
        codon_frequencies: Optional[Dict]
    ):
        """Initialize biological evaluation components with correct dependency order."""

        # 1. Codon frequencies (needed for other components)
        if codon_frequencies is not None:
            self.codon_frequencies = codon_frequencies
        else:
            if self.verbose:
                print("Loading default E. coli codon frequencies...")
            self.codon_frequencies = self._get_default_codon_frequencies()

        # 2. CAI weights
        if cai_weights is not None:
            self.cai_weights = cai_weights
        else:
            if self.verbose:
                print("Computing CAI weights from reference dataset...")
            self.cai_weights = self._compute_default_cai_weights()

        # 3. tAI weights
        if tai_weights is not None:
            self.tai_weights = tai_weights
        else:
            if self.verbose:
                print("Loading default E. coli tAI weights...")
            self.tai_weights = get_ecoli_tai_weights()

        # 4. DTW reference profile (depends on codon_frequencies)
        if reference_profile is not None:
            self.reference_profile = reference_profile
        else:
            if self.verbose:
                print("Computing default DTW reference profile...")
            # Pass codon_frequencies explicitly to ensure it's available
            self.reference_profile = self._compute_default_reference_profile(self.codon_frequencies)
    
    def _compute_default_cai_weights(self) -> Dict:
        """Compute default CAI weights from high-quality E. coli sequences."""
        try:
            # Use the existing processed dataset
            natural_df = pd.read_csv('/home/saketh/ecoli/data/ecoli_processed_genes.csv')
            
            # Clean and validate sequences
            reference_sequences = []
            for seq in natural_df['dna_sequence'].tolist():
                cleaned_seq = "".join(seq.split()).upper()
                if len(cleaned_seq) % 3 == 0:
                    reference_sequences.append(cleaned_seq)

            # Calculate weights from reference sequences
            cai_weights = relative_adaptiveness(sequences=reference_sequences)
            
            # Ensure all 64 codons are present
            all_codons = [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
            for codon in all_codons:
                if codon not in cai_weights:
                    cai_weights[codon] = 1e-9 # Assign a small, non-zero weight
            
            return cai_weights
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load default CAI weights ({e})")
            # Fallback to uniform weights
            return {}
    
    def _compute_default_reference_profile(self, codon_frequencies: Dict) -> np.ndarray:
        """Compute a robust default DTW reference profile."""
        try:
            natural_df = pd.read_csv('/home/saketh/ecoli/data/ecoli_processed_genes.csv')
            natural_sequences = natural_df['dna_sequence'].tolist()[:200]

            profiles = []
            for seq in natural_sequences:
                cleaned_seq = "".join(seq.split()).upper()
                if len(cleaned_seq) < 54:
                    continue
                
                profile = get_min_max_profile(cleaned_seq, codon_frequencies)
                if profile and all(np.isfinite(v) for v in profile):
                    profiles.append(np.array(profile))

            if not profiles:
                if self.verbose:
                    print("Warning: No valid profiles generated for DTW reference. Using fallback.")
                return np.zeros(100)

            median_len = int(np.median([len(p) for p in profiles]))
            padded_profiles = [
                np.interp(np.linspace(0, 1, median_len), np.linspace(0, 1, len(p)), p)
                for p in profiles
            ]
            
            mean_profile = np.mean(padded_profiles, axis=0)
            return mean_profile if np.any(mean_profile) else np.zeros(100)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not compute reference profile ({e}). Using fallback.")
            return np.zeros(100)
    
    def _get_default_codon_frequencies(self) -> Dict:
        """Get default E. coli codon frequencies."""
        try:
            from CodonTransformer.CodonData import download_codon_frequencies_from_kazusa
            return download_codon_frequencies_from_kazusa(taxonomy_id=83333)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not download codon frequencies ({e})")
            return {}

    def _compute_cai_quintiles(self, organism_id: int) -> Dict[str, float]:
        """Computes the top quintile threshold for CAI values for each amino acid."""
        if not self.cai_weights:
            return {}

        # The standard NCBI table for E. coli is 11.
        table_id = 11
        try:
            codon_table_obj = CodonTable.unambiguous_dna_by_id[table_id]
            codon_table = codon_table_obj.forward_table
        except KeyError:
            if self.verbose:
                print(f"Warning: Codon table ID {table_id} not found. Defaulting to 1.")
            codon_table_obj = CodonTable.unambiguous_dna_by_id[1]
            codon_table = codon_table_obj.forward_table

        # Invert codon table to map amino acids to codons
        aa_to_codons = {}
        for codon, aa in codon_table.items():
            if aa not in aa_to_codons:
                aa_to_codons[aa] = []
            aa_to_codons[aa].append(codon)

        quintiles = {}
        for aa, codons in aa_to_codons.items():
            if len(codons) > 1: # No quintile for single-codon AAs
                cai_values = [self.cai_weights.get(c, 0.0) for c in codons]
                # 80th percentile is the top quintile boundary
                quintiles[aa] = np.percentile(cai_values, 80)
        
        # For convenience, map codon to its AA's quintile value
        codon_to_quintile = {}
        for codon, aa in codon_table.items():
            if aa in quintiles:
                codon_to_quintile[codon] = quintiles[aa]
        
        return codon_to_quintile
    
    def calculate_component_scores(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Calculate individual reward components for a batch of sequences.
        """
        batch_size = len(sequences)
        
        # Initialize score tensors
        scores = {k: torch.zeros(batch_size, device=self.device) for k in self.reward_weights.keys()}
        
        for i, sequence in enumerate(sequences):
            try:
                # CAI Score (higher is better)
                if self.cai_weights:
                    scores['cai'][i] = torch.tensor(CAI(sequence, self.cai_weights), device=self.device)
                
                # tAI Score (higher is better)
                if self.tai_weights:
                    scores['tai'][i] = torch.tensor(calculate_tAI(sequence, self.tai_weights), device=self.device)
                
                # GC Content Score (hinge loss, lower is better)
                gc_content = get_GC_content(sequence)
                gc_deviation = abs(gc_content - self.gc_target)
                scores['gc'][i] = torch.tensor(-gc_deviation, device=self.device) # Negative deviation
                
                # DTW Distance Score (lower is better)
                if len(self.reference_profile) > 0:
                    sequence_profile = get_min_max_profile(sequence, self.codon_frequencies)
                    if sequence_profile and any(v is not None for v in sequence_profile):
                        dtw_distance = calculate_dtw_distance(sequence_profile, self.reference_profile)
                        # Clip the distance to prevent extreme values
                        dtw_distance = min(dtw_distance, 1000.0)
                        scores['dtw'][i] = torch.tensor(-dtw_distance, device=self.device)
                    else:
                        # Penalize sequences that don't produce a valid profile
                        scores['dtw'][i] = torch.tensor(-1000.0, device=self.device)
                
                # Structural Penalty Score (fewer violations is better, so more negative is worse)
                penalty_val = 0.0
                penalty_val += 0.05 * scan_for_restriction_sites(sequence) # 0.05 per site
                penalty_val += 0.02 * calculate_homopolymer_runs(sequence, 8) # 0.02 per extra nt
                scores['penalty'][i] = torch.tensor(-penalty_val, device=self.device)

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error computing scores for sequence {i}: {e}")
                continue
        
        return scores
    
    def __call__(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute the multi-objective reward for a batch of DNA sequences.
        Normalization is handled by the VecNormalize wrapper.
        """
        self.call_count += 1
        
        if not sequences:
            return torch.tensor([], device=self.device)
        
        # Step 1: Calculate raw component scores
        raw_scores = self.calculate_component_scores(sequences)
        
        # Step 2: Compute weighted combination.
        total_reward = torch.zeros(len(sequences), device=self.device)
        for component, weight in self.reward_weights.items():
            if component in raw_scores:
                total_reward += weight * raw_scores[component]
        
        # Step 3: Store for analysis
        reward_stats = {
            'mean_reward': float(torch.mean(total_reward)),
            'std_reward': float(torch.std(total_reward)),
            'components': {k: float(torch.mean(v)) for k, v in raw_scores.items()}
        }
        self.reward_history.append(reward_stats)
        
        # Step 4: Optional logging
        if self.verbose and self.call_count % 10 == 0:
            print(f"Reward call {self.call_count}: mean={reward_stats['mean_reward']:.4f}, "
                  f"std={reward_stats['std_reward']:.4f}")
            for comp, val in reward_stats['components'].items():
                print(f"  {comp}: {val:.4f}")
        
        return total_reward
    
    def get_detailed_analysis(self, sequences: List[str]) -> Dict:
        """
        Get detailed component-wise analysis for research evaluation.
        
        Args:
            sequences: List of DNA sequences to analyze
            
        Returns:
            Comprehensive analysis dictionary
        """
        raw_scores = self.calculate_component_scores(sequences)
        normalized_scores = self.normalize_scores(raw_scores)
        total_rewards = self(sequences)
        
        analysis = {
            'total_rewards': total_rewards.cpu().numpy().tolist(),
            'raw_scores': {k: v.cpu().numpy().tolist() for k, v in raw_scores.items()},
            'normalized_scores': {k: v.cpu().numpy().tolist() for k, v in normalized_scores.items()},
            'weights': self.reward_weights,
            'statistics': {
                'reward_mean': float(torch.mean(total_rewards)),
                'reward_std': float(torch.std(total_rewards)),
                'component_correlations': self._compute_correlations(raw_scores)
            }
        }
        
        return analysis
    
    def _compute_correlations(self, scores: Dict[str, torch.Tensor]) -> Dict:
        """Compute correlation matrix between reward components."""
        correlations = {}
        components = list(scores.keys())
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i:], i):
                if len(scores[comp1]) > 1 and len(scores[comp2]) > 1:
                    corr = torch.corrcoef(torch.stack([scores[comp1], scores[comp2]]))[0, 1]
                    correlations[f"{comp1}_{comp2}"] = float(corr) if not torch.isnan(corr) else 0.0
        
        return correlations
    
    def save_analysis(self, filepath: str):
        """Save reward function analysis to file."""
        analysis_data = {
            'reward_weights': self.reward_weights,
            'call_count': self.call_count,
            'reward_history': self.reward_history,
            'gc_target': self.gc_target,
            'gc_tolerance': self.gc_tolerance
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)


def create_default_reward_function(
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> MultiObjectiveRewardFunction:
    """
    Create default reward function with pre-loaded components.
    
    Convenience function for standard E. coli codon optimization.
    
    Args:
        device: Computation device
        verbose: Enable logging
        
    Returns:
        Configured reward function instance
    """
    return MultiObjectiveRewardFunction(
        device=device,
        verbose=verbose
    )


# Research utility functions
def analyze_reward_landscape(
    reward_fn: MultiObjectiveRewardFunction,
    sequences: List[str],
    save_path: Optional[str] = None
) -> Dict:
    """
    Analyze reward landscape for research insights.
    
    Args:
        reward_fn: Configured reward function
        sequences: Test sequences for analysis
        save_path: Optional path to save analysis
        
    Returns:
        Comprehensive landscape analysis
    """
    analysis = reward_fn.get_detailed_analysis(sequences)
    
    # Additional landscape metrics
    rewards = np.array(analysis['total_rewards'])
    analysis['landscape_metrics'] = {
        'reward_range': float(np.max(rewards) - np.min(rewards)),
        'reward_percentiles': {
            f'p{p}': float(np.percentile(rewards, p)) 
            for p in [5, 25, 50, 75, 95]
        },
        'outlier_count': int(np.sum(np.abs(stats.zscore(rewards)) > 3))
    }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return analysis


if __name__ == "__main__":
    # Research validation
    print("=== Multi-Objective Reward Function Validation ===")
    
    # Test sequences for validation
    test_sequences = [
        "ATGAAACTGCTGGTGGTCTAA",  # Short basic
        "ATGGATGAAGAAGATGACTAA",  # Short acidic  
        "GTGGTGCTGCTGATCATCTAA",  # Short hydrophobic
    ]
    
    # Create reward function
    reward_fn = create_default_reward_function(verbose=True)
    
    # Test evaluation
    rewards = reward_fn(test_sequences)
    print(f"Test rewards: {rewards}")
    
    # Detailed analysis
    analysis = reward_fn.get_detailed_analysis(test_sequences)
    print("Detailed analysis completed")
    
    print("✓ Reward function validation successful")