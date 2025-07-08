"""
File: CodonEvaluation.py
---------------------------
Includes functions to calculate various evaluation metrics along with helper
functions.
"""

from typing import Dict, List, Tuple

import pandas as pd
from CAI import CAI, relative_adaptiveness
from tqdm import tqdm
import math
import numpy as np
from collections import Counter
from itertools import chain
from statistics import mean


def get_CSI_weights(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of DNA sequences.

    Args:
        sequences (List[str]): List of DNA sequences.

    Returns:
        dict: The CSI weights.
    """
    return relative_adaptiveness(sequences=sequences)


def get_CSI_value(dna: str, weights: Dict[str, float]) -> float:
    """
    Calculate the Codon Similarity Index (CSI) for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        weights (dict): The CSI weights from get_CSI_weights.

    Returns:
        float: The CSI value.
    """
    return CAI(dna, weights)


def get_organism_to_CSI_weights(
    dataset: pd.DataFrame, organisms: List[str]
) -> Dict[str, dict]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of organisms.

    Args:
        dataset (pd.DataFrame): Dataset containing organism and DNA sequence info.
        organisms (List[str]): List of organism names.

    Returns:
        Dict[str, dict]: A dictionary mapping each organism to its CSI weights.
    """
    organism2weights = {}

    # Iterate through each organism to calculate its CSI weights
    for organism in tqdm(organisms, desc="Calculating CSI Weights: ", unit="Organism"):
        organism_data = dataset.loc[dataset["organism"] == organism]
        sequences = organism_data["dna"].to_list()
        weights = get_CSI_weights(sequences)
        organism2weights[organism] = weights

    return organism2weights


def get_GC_content(dna: str) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        dna (str): The DNA sequence.

    Returns:
        float: The GC content as a percentage.
    """
    dna = dna.upper()
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna) * 100


def get_cfd(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    threshold: float = 0.3,
) -> float:
    """
    Calculate the codon frequency distribution (CFD) metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        threshold (float): Frequency threshold for counting rare codons.

    Returns:
        float: The CFD metric as a percentage.
    """
    # Get a dictionary mapping each codon to its normalized frequency
    codon2frequency = {
        codon: freq / max(frequencies)
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon, freq in zip(codons, frequencies)
    }

    cfd = 0

    # Iterate through the DNA sequence in steps of 3 to process each codon
    for i in range(0, len(dna), 3):
        codon = dna[i : i + 3]
        codon_frequency = codon2frequency[codon]

        if codon_frequency < threshold:
            cfd += 1

    return cfd / (len(dna) / 3) * 100


def get_min_max_percentage(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        window_size (int): Size of the window to calculate %MinMax.

    Returns:
        List[float]: List of %MinMax values for the sequence.

    Credit: https://github.com/chowington/minmax
    """
    # Get a dictionary mapping each codon to its respective amino acid
    codon2amino = {
        codon: amino
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon in codons
    }

    min_max_values = []
    codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]  # Split DNA into codons

    # Iterate through the DNA sequence using the specified window size
    for i in range(len(codons) - window_size + 1):
        codon_window = codons[i : i + window_size]  # Codons in the current window

        Actual = 0.0  # Average of the actual codon frequencies
        Max = 0.0  # Average of the min codon frequencies
        Min = 0.0  # Average of the max codon frequencies
        Avg = 0.0  # Average of the averages of all frequencies for each amino acid

        # Sum the frequencies for codons in the current window
        for codon in codon_window:
            aminoacid = codon2amino[codon]
            frequencies = codon_frequencies[aminoacid][1]
            codon_index = codon_frequencies[aminoacid][0].index(codon)
            codon_frequency = codon_frequencies[aminoacid][1][codon_index]

            Actual += codon_frequency
            Max += max(frequencies)
            Min += min(frequencies)
            Avg += sum(frequencies) / len(frequencies)

        # Divide by the window size to get the averages
        Actual = Actual / window_size
        Max = Max / window_size
        Min = Min / window_size
        Avg = Avg / window_size

        # Calculate %MinMax
        percentMax = ((Actual - Avg) / (Max - Avg)) * 100
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100

        # Append the appropriate %MinMax value
        if percentMax >= 0:
            min_max_values.append(percentMax)
        else:
            min_max_values.append(-percentMin)

    # Populate the last floor(window_size / 2) entries of min_max_values with None
    for i in range(int(window_size / 2)):
        min_max_values.append(None)

    return min_max_values


def get_sequence_complexity(dna: str) -> float:
    """
    Calculate the sequence complexity score of a DNA sequence.

    Args:
        dna (str): The DNA sequence.

    Returns:
        float: The sequence complexity score.
    """

    def sum_up_to(x):
        """Recursive function to calculate the sum of integers from 1 to x."""
        if x <= 1:
            return 1
        else:
            return x + sum_up_to(x - 1)

    def f(x):
        """Returns 4 if x is greater than or equal to 4, else returns x."""
        if x >= 4:
            return 4
        elif x < 4:
            return x

    unique_subseq_length = []

    # Calculate unique subsequences lengths
    for i in range(1, len(dna) + 1):
        unique_subseq = set()
        for j in range(len(dna) - (i - 1)):
            unique_subseq.add(dna[j : (j + i)])
        unique_subseq_length.append(len(unique_subseq))

    # Calculate complexity score
    complexity_score = (
        sum(unique_subseq_length) / (sum_up_to(len(dna) - 1) + f(len(dna)))
    ) * 100

    return complexity_score


def get_sequence_similarity(
    original: str, predicted: str, truncate: bool = True, window_length: int = 1
) -> float:
    """
    Calculate the sequence similarity between two sequences.

    Args:
        original (str): The original sequence.
        predicted (str): The predicted sequence.
        truncate (bool): If True, truncate the original sequence to match the length
            of the predicted sequence.
        window_length (int): Length of the window for comparison (1 for amino acids,
            3 for codons).

    Returns:
        float: The sequence similarity as a percentage.

    Preconditions:
        len(predicted) <= len(original).
    """
    if not truncate and len(original) != len(predicted):
        raise ValueError(
            "Set truncate to True if the length of sequences do not match."
        )

    identity = 0.0
    original = original.strip()
    predicted = predicted.strip()

    if truncate:
        original = original[: len(predicted)]

    if window_length == 1:
        # Simple comparison for amino acid
        for i in range(len(predicted)):
            if original[i] == predicted[i]:
                identity += 1
    else:
        # Comparison for substrings based on window_length
        for i in range(0, len(original) - window_length + 1, window_length):
            if original[i : i + window_length] == predicted[i : i + window_length]:
                identity += 1

    return (identity / (len(predicted) / window_length)) * 100


def scan_for_restriction_sites(seq: str, sites: List[str] = ['GAATTC', 'GGATCC', 'AAGCTT']) -> int:
    """
    Scans for a list of restriction enzyme sites in a DNA sequence.
    """
    return sum(seq.upper().count(site.upper()) for site in sites)


def count_negative_cis_elements(seq: str, motifs: List[str] = ['TATAAT', 'TTGACA', 'AGCTAGT']) -> int:
    """
    Counts occurrences of negative cis-regulatory elements in a DNA sequence.
    """
    return sum(seq.upper().count(m.upper()) for m in motifs)


def calculate_homopolymer_runs(seq: str, max_len: int = 8) -> int:
    """
    Calculates the number of homopolymer runs longer than a given length.
    """
    import re
    min_len = max_len + 1
    return len(re.findall(r'(A{%d,}|T{%d,}|G{%d,}|C{%d,})' % (min_len, min_len, min_len, min_len), seq.upper()))


def get_min_max_profile(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax profile for a DNA sequence. This is a list of
    %MinMax values for sliding windows across the sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        window_size (int): Size of the window to calculate %MinMax.

    Returns:
        List[float]: List of %MinMax values for the sequence.
    """
    return get_min_max_percentage(dna, codon_frequencies, window_size)


def calculate_dtw_distance(profile1: List[float], profile2: List[float]) -> float:
    """
    Calculates the Dynamic Time Warping (DTW) distance between two profiles.

    Args:
        profile1 (List[float]): The first profile (e.g., %MinMax of generated sequence).
        profile2 (List[float]): The second profile (e.g., %MinMax of natural sequence).

    Returns:
        float: The DTW distance between the two profiles.
    """
    from dtw import dtw
    import numpy as np

    # Ensure profiles are numpy arrays and handle potential None and NaN values
    p1 = np.array([v for v in profile1 if v is not None and not np.isnan(v)]).reshape(
        -1, 1
    )
    p2 = np.array([v for v in profile2 if v is not None and not np.isnan(v)]).reshape(
        -1, 1
    )

    if len(p1) == 0 or len(p2) == 0:
        return np.inf  # Return infinity if one of the profiles is empty

    alignment = dtw(p1, p2, keep_internals=True)
    return alignment.distance  # type: ignore


def get_ecoli_tai_weights():
    """
    Returns a dictionary of tAI weights for E. coli based on tRNA gene copy numbers.
    These weights are pre-calculated based on the relative adaptiveness of each codon.
    """
    codons = [
        "TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG", "TAT", "TAC",
        "TGT", "TGC", "TGG", "CTT", "CTC", "CTA", "CTG", "CCT", "CCC", "CCA",
        "CCG", "CAT", "CAC", "CAA", "CAG", "CGT", "CGC", "CGA", "CGG", "ATT",
        "ATC", "ATA", "ACT", "ACC", "ACA", "ACG", "AAT", "AAC", "AAA", "AAG",
        "AGT", "AGC", "AGA", "AGG", "GTT", "GTC", "GTA", "GTG", "GCT", "GCC",
        "GCA", "GCG", "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA", "GGG"
    ]
    weights = [
        0.1966667, 0.3333333, 0.1666667, 0.2200000, 0.1966667, 0.3333333,
        0.1666667, 0.2200000, 0.2950000, 0.5000000, 0.09833333, 0.1666667,
        0.2200000, 0.09833333, 0.1666667, 0.1666667, 0.7200000, 0.09833333,
        0.1666667, 0.1666667, 0.2200000, 0.09833333, 0.1666667, 0.3333333,
        0.4400000, 0.6666667, 0.4800000, 0.00006666667, 0.1666667, 0.2950000,
        0.5000000, 0.01833333, 0.1966667, 0.3333333, 0.1666667, 0.3866667,
        0.3933333, 0.6666667, 1.0000000, 0.3200000, 0.09833333, 0.1666667,
        0.1666667, 0.2200000, 0.1966667, 0.3333333, 0.8333333, 0.2666667,
        0.1966667, 0.3333333, 0.5000000, 0.1600000, 0.2950000, 0.5000000,
        0.6666667, 0.2133333, 0.3933333, 0.6666667, 0.1666667, 0.2200000
    ]
    return dict(zip(codons, weights))


def calculate_tAI(sequence: str, tai_weights: Dict[str, float]) -> float:
    """
    Calculates the tRNA Adaptation Index (tAI) for a given DNA sequence.

    Args:
        sequence (str): The DNA sequence to analyze.
        tai_weights (Dict[str, float]): A dictionary of tAI weights for each codon.

    Returns:
        float: The tAI value for the sequence.
    """
    from scipy.stats.mstats import gmean
    
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # Filter out stop codons and codons not in weights
    weights = [tai_weights[codon] for codon in codons if codon in tai_weights and tai_weights[codon] > 0]
    
    if not weights:
        return 0.0
        
    return gmean(weights)


def calculate_ENC(sequence: str) -> float:
    """
    Calculate the Effective Number of Codons (ENC) for a DNA sequence.
    Based on Wright (1990) and Fuglsang (2004) corrections.
    
    Args:
        sequence (str): The DNA sequence.
    
    Returns:
        float: The ENC value for the sequence.
    """
    # Genetic code mapping (standard genetic code)
    genetic_code = {
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
    
    # Amino acid to codon mapping
    aa_to_codons = {
        'F': ['TTT', 'TTC'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'Y': ['TAT', 'TAC'],
        'C': ['TGT', 'TGC'],
        'W': ['TGG'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'H': ['CAT', 'CAC'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'I': ['ATT', 'ATC', 'ATA'],
        'M': ['ATG'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'N': ['AAT', 'AAC'],
        'K': ['AAA', 'AAG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG']
    }
    
    # Count codons in sequence
    codons = [sequence[i:i+3].upper() for i in range(0, len(sequence), 3)]
    codon_counts = Counter(codons)
    
    # Group amino acids by degeneracy
    sf_groups = {
        1: ['M', 'W'],  # non-degenerate
        2: ['F', 'Y', 'C', 'H', 'Q', 'N', 'K', 'D', 'E'],  # 2-fold degenerate
        3: ['I'],  # 3-fold degenerate  
        4: ['P', 'T', 'A', 'V', 'G'],  # 4-fold degenerate
        6: ['L', 'S', 'R']  # 6-fold degenerate
    }
    
    # Calculate F values for each degeneracy group
    F_values = {}
    
    for sf, amino_acids in sf_groups.items():
        if sf == 1:
            F_values[sf] = 1.0  # Non-degenerate amino acids
            continue
            
        group_f_values = []
        for aa in amino_acids:
            if aa not in aa_to_codons:
                continue
                
            codons_for_aa = aa_to_codons[aa]
            counts = [codon_counts.get(codon, 0) for codon in codons_for_aa]
            total_count = sum(counts)
            
            if total_count == 0:
                continue
                
            # Calculate F value for this amino acid
            p_squared_sum = sum((count / total_count) ** 2 for count in counts)
            
            if total_count == 1:
                f_value = 1.0
            else:
                f_value = (total_count * p_squared_sum - 1) / (total_count - 1)
                
            group_f_values.append(f_value)
        
        if group_f_values:
            F_values[sf] = mean(group_f_values)
        else:
            F_values[sf] = 1.0
    
    # Handle missing 3-fold degenerate (Fuglsang correction)
    if 3 not in F_values or F_values[3] == 0:
        f_2 = F_values.get(2, 1.0)
        f_4 = F_values.get(4, 1.0)
        f_6 = F_values.get(6, 1.0)
        
        if f_2 > 0 and f_4 > 0 and f_6 > 0:
            F_values[3] = ((2/f_2 - 1)**-1 + (2/(3*f_4) + 1/3)**-1 + (2/(5*f_6) + 3/5)**-1) / 3.0
        else:
            F_values[3] = 1.0
    
    # Calculate ENC
    enc_value = (
        2 +  # M, W (1-fold)
        (9 / F_values.get(2, 1.0)) +  # 2-fold degenerate
        (1 / F_values.get(3, 1.0)) +  # 3-fold degenerate (I)
        (5 / F_values.get(4, 1.0)) +  # 4-fold degenerate
        (3 / F_values.get(6, 1.0))    # 6-fold degenerate
    )
    
    return min(enc_value, 61.0)


def calculate_CPB(sequence: str, reference_sequences: List[str] = None) -> float:
    """
    Calculate the Codon Pair Bias (CPB) for a DNA sequence.
    Based on Gutman & Hatfield (1989) and Coleman et al. (2008).
    
    Args:
        sequence (str): The DNA sequence.
        reference_sequences (List[str]): Reference sequences for calculating expected values.
                                       If None, uses the input sequence itself.
    
    Returns:
        float: The CPB value for the sequence.
    """
    if reference_sequences is None:
        reference_sequences = [sequence]
    
    # Calculate codon pair frequencies in reference set
    all_codons = []
    all_pairs = []
    
    for ref_seq in reference_sequences:
        codons = [ref_seq[i:i+3].upper() for i in range(0, len(ref_seq), 3)]
        all_codons.extend(codons)
        
        # Extract codon pairs
        for i in range(len(codons) - 1):
            all_pairs.append((codons[i], codons[i+1]))
    
    # Count individual codons and codon pairs
    codon_counts = Counter(all_codons)
    pair_counts = Counter(all_pairs)
    
    total_codons = len(all_codons)
    total_pairs = len(all_pairs)
    
    if total_pairs == 0:
        return 0.0
    
    # Calculate CPB for the target sequence
    target_codons = [sequence[i:i+3].upper() for i in range(0, len(sequence), 3)]
    target_pairs = [(target_codons[i], target_codons[i+1]) for i in range(len(target_codons) - 1)]
    
    if not target_pairs:
        return 0.0
    
    cpb_sum = 0.0
    valid_pairs = 0
    
    for pair in target_pairs:
        codon1, codon2 = pair
        
        # Observed frequency of this pair
        observed = pair_counts.get(pair, 0) / total_pairs if total_pairs > 0 else 0
        
        # Expected frequency (product of individual codon frequencies)
        freq1 = codon_counts.get(codon1, 0) / total_codons if total_codons > 0 else 0
        freq2 = codon_counts.get(codon2, 0) / total_codons if total_codons > 0 else 0
        expected = freq1 * freq2
        
        if expected > 0:
            cpb_sum += math.log(observed / expected) if observed > 0 else math.log(1e-10 / expected)
            valid_pairs += 1
    
    return cpb_sum / valid_pairs if valid_pairs > 0 else 0.0


def calculate_SCUO(sequence: str) -> float:
    """
    Calculate the Synonymous Codon Usage Order (SCUO) for a DNA sequence.
    Based on Wan et al. (2004) - measures deviation from random codon usage.
    
    Args:
        sequence (str): The DNA sequence.
    
    Returns:
        float: The SCUO value (0-1, where 1 indicates maximum bias).
    """
    # Amino acid to codon mapping (same as in ENC calculation)
    aa_to_codons = {
        'F': ['TTT', 'TTC'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'Y': ['TAT', 'TAC'],
        'C': ['TGT', 'TGC'],
        'W': ['TGG'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'H': ['CAT', 'CAC'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'I': ['ATT', 'ATC', 'ATA'],
        'M': ['ATG'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'N': ['AAT', 'AAC'],
        'K': ['AAA', 'AAG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG']
    }
    
    # Genetic code mapping
    genetic_code = {
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
    
    # Count codons
    codons = [sequence[i:i+3].upper() for i in range(0, len(sequence), 3)]
    codon_counts = Counter(codons)
    
    scuo_values = []
    total_weight = 0
    
    # Calculate SCUO for each amino acid with synonymous codons
    for aa, aa_codons in aa_to_codons.items():
        if len(aa_codons) <= 1:  # Skip non-degenerate amino acids
            continue
            
        # Count usage of each codon for this amino acid
        aa_codon_counts = [codon_counts.get(codon, 0) for codon in aa_codons]
        total_aa_count = sum(aa_codon_counts)
        
        if total_aa_count == 0:
            continue
            
        # Calculate frequencies
        frequencies = [count / total_aa_count for count in aa_codon_counts]
        
        # Calculate entropy
        entropy = -sum(f * math.log(f) if f > 0 else 0 for f in frequencies)
        
        # Maximum possible entropy for this amino acid
        max_entropy = math.log(len(aa_codons))
        
        # Calculate SCUO for this amino acid (1 - normalized entropy)
        if max_entropy > 0:
            aa_scuo = 1.0 - (entropy / max_entropy)
        else:
            aa_scuo = 0.0
            
        scuo_values.append(aa_scuo * total_aa_count)
        total_weight += total_aa_count
    
    # Calculate weighted average SCUO
    if total_weight > 0:
        return sum(scuo_values) / total_weight
    else:
        return 0.0
