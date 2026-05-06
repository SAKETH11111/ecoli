import math
from collections import Counter
from itertools import chain


_FORWARD_TABLE_11 = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TGT": "C", "TGC": "C", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_STOP_CODONS_11 = {"TAA", "TAG", "TGA"}


def _build_synonymous_codons(forward_table):
    codons_for_amino_acid = {}
    for codon, amino_acid in forward_table.items():
        codons_for_amino_acid.setdefault(amino_acid, []).append(codon)
    return {
        codon: codons_for_amino_acid[forward_table[codon]]
        for codon in forward_table
    }


_SYNONYMOUS_CODONS_11 = _build_synonymous_codons(_FORWARD_TABLE_11)
_NON_SYNONYMOUS_CODONS_11 = {
    codon for codon, group in _SYNONYMOUS_CODONS_11.items() if len(group) == 1
}


def _require_genetic_code_11(genetic_code):
    if genetic_code != 11:
        raise NotImplementedError("This bundled CAI fallback currently supports only genetic code 11.")


def _geometric_mean(values):
    if not values:
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def RSCU(sequences, genetic_code=11):
    _require_genetic_code_11(genetic_code)

    if not isinstance(sequences, (list, tuple)):
        raise ValueError(
            "Be sure to pass a list of sequences, not a single sequence. "
            "To find the RSCU of a single sequence, pass it as a one element list."
        )

    for sequence in sequences:
        if not sequence:
            raise ValueError("Input sequence cannot be empty")
        if len(sequence) % 3 != 0:
            raise ValueError("Input sequence not divisible by three")

    codon_streams = (
        (sequence[i : i + 3].upper() for i in range(0, len(sequence), 3))
        for sequence in sequences
    )
    counts = Counter(chain.from_iterable(codon_streams))

    for codon in _FORWARD_TABLE_11:
        if counts[codon] == 0:
            counts[codon] = 0.5

    result = {}
    for codon in _FORWARD_TABLE_11:
        codon_group = _SYNONYMOUS_CODONS_11[codon]
        result[codon] = counts[codon] / (
            (len(codon_group) ** -1) * sum(counts[group_codon] for group_codon in codon_group)
        )
    return result


def relative_adaptiveness(sequences=None, RSCUs=None, genetic_code=11):
    _require_genetic_code_11(genetic_code)

    if sum([bool(sequences), bool(RSCUs)]) != 1:
        raise TypeError("Must provide either reference sequences or RSCU dictionary")

    if sequences:
        RSCUs = RSCU(sequences, genetic_code=genetic_code)

    return {
        codon: value / max(RSCUs[group_codon] for group_codon in _SYNONYMOUS_CODONS_11[codon])
        for codon, value in RSCUs.items()
    }


def CAI(sequence, weights=None, RSCUs=None, reference=None, genetic_code=11):
    _require_genetic_code_11(genetic_code)

    if sum([bool(reference), bool(RSCUs), bool(weights)]) != 1:
        raise TypeError(
            "Must provide either reference sequences, or RSCU dictionary, or weights"
        )
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    if len(sequence) % 3 != 0:
        raise ValueError("Input sequence not divisible by three")

    sequence = sequence.upper()
    codons = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]

    if reference:
        weights = relative_adaptiveness(sequences=reference, genetic_code=genetic_code)
    elif RSCUs:
        weights = relative_adaptiveness(RSCUs=RSCUs, genetic_code=genetic_code)

    sequence_weights = []
    for codon in codons:
        if codon in _NON_SYNONYMOUS_CODONS_11 or codon in _STOP_CODONS_11:
            continue
        if codon not in weights:
            raise KeyError(
                "Bad weights dictionary passed: missing weight for codon "
                + str(codon)
                + "."
            )
        sequence_weights.append(weights[codon])

    return float(_geometric_mean(sequence_weights))
