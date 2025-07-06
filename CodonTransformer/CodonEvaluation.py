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
    Calculate the %MinMax metric for a DNA sequence with robust error handling.
    """
    codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]
    if not codons:
        return []

    # Return a neutral profile for sequences shorter than the window size
    if len(codons) < window_size:
        return [0.0] * len(codons)

    codon2amino = {
        codon: amino
        for amino, (codons_list, frequencies) in codon_frequencies.items()
        for codon in codons_list
    }

    min_max_values = []

    for i in range(len(codons) - window_size + 1):
        codon_window = codons[i : i + window_size]
        
        Actual, Max, Min, Avg = 0.0, 0.0, 0.0, 0.0
        valid_codons_in_window = 0

        for codon in codon_window:
            aminoacid = codon2amino.get(codon)
            if not aminoacid or aminoacid not in codon_frequencies:
                continue

            frequencies = codon_frequencies[aminoacid][1]
            if not frequencies:
                continue

            try:
                codon_index = codon_frequencies[aminoacid][0].index(codon)
                codon_frequency = frequencies[codon_index]
            except ValueError:
                continue
            
            valid_codons_in_window += 1
            Actual += codon_frequency
            Max += max(frequencies)
            Min += min(frequencies)
            Avg += sum(frequencies) / len(frequencies)

        if valid_codons_in_window == 0:
            min_max_values.append(0.0)
            continue

        Actual /= valid_codons_in_window
        Max /= valid_codons_in_window
        Min /= valid_codons_in_window
        Avg /= valid_codons_in_window

        percentMax = ((Actual - Avg) / (Max - Avg)) * 100 if (Max - Avg) != 0 else 0.0
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100 if (Avg - Min) != 0 else 0.0

        if percentMax >= 0:
            min_max_values.append(percentMax)
        else:
            min_max_values.append(-percentMin)

    # Pad the result to match the original codon length, ensuring a full profile
    padding_size = len(codons) - len(min_max_values)
    if padding_size > 0:
        min_max_values.extend([0.0] * padding_size)

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
    Calculates the total number of extra nucleotides in homopolymer runs longer than max_len.
    For example, for max_len=8, a run of 10 'A's contributes 2 to the count.
    """
    import re
    total_extra_nt = 0
    min_len = max_len + 1
    # Regex to find any character repeated more than max_len times
    for match in re.finditer(r'(A{%d,}|T{%d,}|G{%d,}|C{%d,})' % (min_len, min_len, min_len, min_len), seq.upper()):
        # The length of the found run minus the allowed length
        total_extra_nt += len(match.group(0)) - max_len
    return total_extra_nt


def get_min_max_profile(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax profile for a DNA sequence with robust error handling.
    """
    return get_min_max_percentage(dna, codon_frequencies, window_size)


def calculate_dtw_distance(profile1: List[float], profile2: List[float]) -> float:
    """
    Calculates the Dynamic Time Warping (DTW) distance between two profiles
    with enhanced stability and error handling.
    """
    from dtw import dtw
    import numpy as np

    p1 = np.array([v for v in profile1 if v is not None and np.isfinite(v)])
    p2 = np.array([v for v in profile2 if v is not None and np.isfinite(v)])

    if len(p1) < 2 or len(p2) < 2:
        return 0.0  # Return neutral distance for invalid profiles

    # Resample the shorter sequence to match the longer one
    if len(p1) != len(p2):
        if len(p1) < len(p2):
            p1 = np.interp(np.linspace(0, 1, len(p2)), np.linspace(0, 1, len(p1)), p1)
        else:
            p2 = np.interp(np.linspace(0, 1, len(p1)), np.linspace(0, 1, len(p2)), p2)

    # Normalize profiles to have zero mean and unit variance
    p1 = (p1 - np.mean(p1)) / (np.std(p1) + 1e-9)
    p2 = (p2 - np.mean(p2)) / (np.std(p2) + 1e-9)

    try:
        alignment = dtw(p1, p2, keep_internals=True)
        return alignment.distance
    except Exception:
        return 1000.0 # Return a high penalty if DTW fails


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
