"""
File: CodonPrediction.py
---------------------------
Includes functions to tokenize input, load models, infer predicted dna sequences and
helper functions related to processing data for passing to the model.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as rt
import torch
import transformers
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BigBirdConfig,
    BigBirdForMaskedLM,
    PreTrainedTokenizerFast,
)

from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonUtils import (
    AMINO_ACID_TO_INDEX,
    INDEX2TOKEN,
    NUM_ORGANISMS,
    ORGANISM2ID,
    TOKEN2INDEX,
    DNASequencePrediction,
    CODON_GC_CONTENT,
)


def predict_dna_sequence(
    protein: str,
    organism: Union[int, str],
    device: torch.device,
    tokenizer: Union[str, PreTrainedTokenizerFast] = None,
    model: Union[str, torch.nn.Module] = None,
    attention_type: str = "original_full",
    deterministic: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_sequences: int = 1,
    match_protein: bool = False,
) -> Union[DNASequencePrediction, List[DNASequencePrediction]]:
    """
    Predict the DNA sequence(s) for a given protein using the CodonTransformer model.

    This function takes a protein sequence and an organism (as ID or name) as input
    and returns the predicted DNA sequence(s) using the CodonTransformer model. It can use
    either provided tokenizer and model objects or load them from specified paths.

    Args:
        protein (str): The input protein sequence for which to predict the DNA sequence.
        organism (Union[int, str]): Either the ID of the organism or its name (e.g.,
            "Escherichia coli general"). If a string is provided, it will be converted
            to the corresponding ID using ORGANISM2ID.
        device (torch.device): The device (CPU or GPU) to run the model on.
        tokenizer (Union[str, PreTrainedTokenizerFast, None], optional): Either a file
            path to load the tokenizer from, a pre-loaded tokenizer object, or None. If
            None, it will be loaded from HuggingFace. Defaults to None.
        model (Union[str, torch.nn.Module, None], optional): Either a file path to load
            the model from, a pre-loaded model object, or None. If None, it will be
            loaded from HuggingFace. Defaults to None.
        attention_type (str, optional): The type of attention mechanism to use in the
            model. Can be either 'block_sparse' or 'original_full'. Defaults to
            "original_full".
        deterministic (bool, optional): Whether to use deterministic decoding (most
            likely tokens). If False, samples tokens according to their probabilities
            adjusted by the temperature. Defaults to True.
        temperature (float, optional): A value controlling the randomness of predictions
            during non-deterministic decoding. Lower values (e.g., 0.2) make the model
            more conservative, while higher values (e.g., 0.8) increase randomness.
            Using high temperatures may result in prediction of DNA sequences that
            do not translate to the input protein.
            Recommended values are:
                - Low randomness: 0.2
                - Medium randomness: 0.5
                - High randomness: 0.8
            The temperature must be a positive float. Defaults to 0.2.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling.
            Tokens with cumulative probability up to top_p are considered for sampling.
            This parameter helps balance diversity and coherence in the predicted DNA sequences.
            The value must be a float between 0 and 1. Defaults to 0.95.
        num_sequences (int, optional): The number of DNA sequences to generate. Only applicable
            when deterministic is False. Defaults to 1.
        match_protein (bool, optional): Ensures the predicted DNA sequence is translated
            to the input protein sequence by sampling from only the respective codons of
            given amino acids. Defaults to False.

    Returns:
        Union[DNASequencePrediction, List[DNASequencePrediction]]: An object or list of objects
        containing the prediction results:
            - organism (str): Name of the organism used for prediction.
            - protein (str): Input protein sequence for which DNA sequence is predicted.
            - processed_input (str): Processed input sequence (merged protein and DNA).
            - predicted_dna (str): Predicted DNA sequence.

    Raises:
        ValueError: If the protein sequence is empty, if the organism is invalid,
            if the temperature is not a positive float, if top_p is not between 0 and 1,
            or if num_sequences is less than 1 or used with deterministic mode.

    Note:
        This function uses ORGANISM2ID, INDEX2TOKEN, and AMINO_ACID_TO_INDEX dictionaries
        imported from CodonTransformer.CodonUtils. ORGANISM2ID maps organism names to their
        corresponding IDs. INDEX2TOKEN maps model output indices (token IDs) to
        respective codons. AMINO_ACID_TO_INDEX maps each amino acid and stop symbol to indices
        of codon tokens that translate to it.

    Example:
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForMaskedLM
        >>> from CodonTransformer.CodonPrediction import predict_dna_sequence
        >>> from CodonTransformer.CodonJupyter import format_model_output
        >>>
        >>> # Set up device
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>>
        >>> # Load tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        >>> model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        >>> model = model.to(device)
        >>>
        >>> # Define protein sequence and organism
        >>> protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"
        >>> organism = "Escherichia coli general"
        >>>
        >>> # Predict DNA sequence with deterministic decoding (single sequence)
        >>> output = predict_dna_sequence(
        ...     protein=protein,
        ...     organism=organism,
        ...     device=device,
        ...     tokenizer=tokenizer,
        ...     model=model,
        ...     attention_type="original_full",
        ...     deterministic=True
        ... )
        >>>
        >>> # Predict multiple DNA sequences with low randomness and top_p sampling
        >>> output_random = predict_dna_sequence(
        ...     protein=protein,
        ...     organism=organism,
        ...     device=device,
        ...     tokenizer=tokenizer,
        ...     model=model,
        ...     attention_type="original_full",
        ...     deterministic=False,
        ...     temperature=0.2,
        ...     top_p=0.95,
        ...     num_sequences=3
        ... )
        >>>
        >>> print(format_model_output(output))
        >>> for i, seq in enumerate(output_random, 1):
        ...     print(f"Sequence {i}:")
        ...     print(format_model_output(seq))
        ...     print()
    """
    if not protein:
        raise ValueError("Protein sequence cannot be empty.")

    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("Temperature must be a positive float.")

    if not isinstance(top_p, (float, int)) or not 0 < top_p <= 1.0:
        raise ValueError("top_p must be a float between 0 and 1.")

    if not isinstance(num_sequences, int) or num_sequences < 1:
        raise ValueError("num_sequences must be a positive integer.")

    if deterministic and num_sequences > 1:
        raise ValueError(
            "Multiple sequences can only be generated in non-deterministic mode."
        )

    # Load tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = load_tokenizer(tokenizer)

    # Load model
    if not isinstance(model, torch.nn.Module):
        model = load_model(model_path=model, device=device, attention_type=attention_type)
    else:
        model.eval()
        model.bert.set_attention_type(attention_type)
        model.to(device)

    # Validate organism and convert to organism_id and organism_name
    organism_id, organism_name = validate_and_convert_organism(organism)

    # Inference loop
    with torch.no_grad():
        # Tokenize the input sequence
        merged_seq = get_merged_seq(protein=protein, dna="")
        input_dict = {
            "idx": 0,  # sample index
            "codons": merged_seq,
            "organism": organism_id,
        }
        tokenized_input = tokenize([input_dict], tokenizer=tokenizer).to(device)

        # Get the model predictions
        output_dict = model(**tokenized_input, return_dict=True)
        logits = output_dict.logits.detach().cpu()
        logits = logits[:, 1:-1, :]  # Remove [CLS] and [SEP] tokens

        # Mask the logits of codons that do not correspond to the input protein sequence
        if match_protein:
            possible_tokens_per_position = [
                AMINO_ACID_TO_INDEX[token[0]] for token in merged_seq.split(" ")
            ]
            seq_len = logits.shape[1]
            if len(possible_tokens_per_position) > seq_len:
                possible_tokens_per_position = possible_tokens_per_position[:seq_len]

            mask = torch.full_like(logits, float("-inf"))

            for pos, possible_tokens in enumerate(possible_tokens_per_position):
                mask[:, pos, possible_tokens] = 0

            logits = mask + logits

        predictions = []
        for _ in range(num_sequences):
            # Decode the predicted DNA sequence from the model output
            if deterministic:
                predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
            else:
                predicted_indices = sample_non_deterministic(
                    logits=logits, temperature=temperature, top_p=top_p
                )

            predicted_dna = list(map(INDEX2TOKEN.__getitem__, predicted_indices))
            predicted_dna = (
                "".join([token[-3:] for token in predicted_dna]).strip().upper()
            )

            predictions.append(
                DNASequencePrediction(
                    organism=organism_name,
                    protein=protein,
                    processed_input=merged_seq,
                    predicted_dna=predicted_dna,
                )
            )

    return predictions[0] if num_sequences == 1 else predictions


def sample_non_deterministic(
    logits: torch.Tensor,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> List[int]:
    """
    Sample token indices from logits using temperature scaling and nucleus (top-p) sampling.

    This function applies temperature scaling to the logits, computes probabilities,
    and then performs nucleus sampling to select token indices. It is used for
    non-deterministic decoding in language models to introduce randomness while
    maintaining coherence in the generated sequences.

    Args:
        logits (torch.Tensor): The logits output from the model of shape
            [seq_len, vocab_size] or [batch_size, seq_len, vocab_size].
        temperature (float, optional): Temperature value for scaling logits.
            Must be a positive float. Defaults to 1.0.
        top_p (float, optional): Cumulative probability threshold for nucleus sampling.
            Must be a float between 0 and 1. Tokens with cumulative probability up to
            `top_p` are considered for sampling. Defaults to 0.95.

    Returns:
        List[int]: A list of sampled token indices corresponding to the predicted tokens.

    Raises:
        ValueError: If `temperature` is not a positive float or if `top_p` is not between 0 and 1.

    Example:
        >>> logits = model_output.logits  # Assume logits is a tensor of shape [seq_len, vocab_size]
        >>> predicted_indices = sample_non_deterministic(logits, temperature=0.7, top_p=0.9)
    """
    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("Temperature must be a positive float.")

    if not isinstance(top_p, (float, int)) or not 0 < top_p <= 1.0:
        raise ValueError("top_p must be a float between 0 and 1.")

    # Compute probabilities using temperature scaling
    probs = torch.softmax(logits / temperature, dim=-1)


    # Remove batch dimension if present
    if probs.dim() == 3:
        probs = probs.squeeze(0)  # Shape: [seq_len, vocab_size]

    # Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p

    # Zero out probabilities for tokens beyond the top-p threshold
    probs_sort[mask] = 0.0

    # Renormalize the probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    predicted_indices = torch.gather(probs_idx, -1, next_token).squeeze(-1)

    return predicted_indices.tolist()


def load_model(
    model_path: Optional[str] = None,
    device: torch.device = None,
    attention_type: str = "original_full",
    num_organisms: int = None,
    remove_prefix: bool = True,
) -> torch.nn.Module:
    """
    Load a BigBirdForMaskedLM model from a model file, checkpoint, or HuggingFace.

    Args:
        model_path (Optional[str]): Path to the model file or checkpoint. If None,
            load from HuggingFace.
        device (torch.device, optional): The device to load the model onto.
        attention_type (str, optional): The type of attention, 'block_sparse'
            or 'original_full'.
        num_organisms (int, optional): Number of organisms, needed if loading from a
            checkpoint that requires this.
        remove_prefix (bool, optional): Whether to remove the "model." prefix from the
            keys in the state dict.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if not model_path:
        warnings.warn("Model path not provided. Loading from HuggingFace.", UserWarning)
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
    elif model_path.endswith(".ckpt"):
        checkpoint = torch.load(model_path, map_location="cpu")

        # Detect Lightning checkpoint vs raw state dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            if remove_prefix:
                state_dict = {
                    k.replace("model.", ""): v for k, v in state_dict.items()
                }
        else:
            # assume checkpoint itself is state_dict
            state_dict = checkpoint

        if num_organisms is None:
            num_organisms = NUM_ORGANISMS

        # Load model configuration and instantiate the model
        config = load_bigbird_config(num_organisms)
        model = BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict, strict=False)

    elif model_path.endswith(".pt"):
        state_dict = torch.load(model_path)
        config = state_dict.pop("self.config")
        model = BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(
            "Unsupported file type. Please provide a .ckpt or .pt file, "
            "or None to load from HuggingFace."
        )

    # Prepare model for evaluation
    model.bert.set_attention_type(attention_type)
    model.eval()
    if device:
        model.to(device)

    return model


def load_bigbird_config(num_organisms: int) -> BigBirdConfig:
    """
    Load the config object used to train the BigBird transformer.

    Args:
        num_organisms (int): The number of organisms.

    Returns:
        BigBirdConfig: The configuration object for BigBird.
    """
    config = transformers.BigBirdConfig(
        vocab_size=len(TOKEN2INDEX),  # Equal to len(tokenizer)
        type_vocab_size=num_organisms,
        sep_token_id=2,
    )
    return config


def create_model_from_checkpoint(
    checkpoint_dir: str, output_model_dir: str, num_organisms: int
) -> None:
    """
    Save a model to disk using a previous checkpoint.

    Args:
        checkpoint_dir (str): Directory where the checkpoint is stored.
        output_model_dir (str): Directory where the model will be saved.
        num_organisms (int): Number of organisms.
    """
    checkpoint = load_model(model_path=checkpoint_dir, num_organisms=num_organisms)
    state_dict = checkpoint.state_dict()
    state_dict["self.config"] = load_bigbird_config(num_organisms=num_organisms)

    # Save the model state dict to the output directory
    torch.save(state_dict, output_model_dir)


def load_tokenizer(tokenizer_path: Optional[str] = None) -> PreTrainedTokenizerFast:
    """
    Create and return a tokenizer object from tokenizer path or HuggingFace.

    Args:
        tokenizer_path (Optional[str]): Path to the tokenizer file. If None,
        load from HuggingFace.

    Returns:
        PreTrainedTokenizerFast: The tokenizer object.
    """
    if not tokenizer_path:
        warnings.warn(
            "Tokenizer path not provided. Loading from HuggingFace.", UserWarning
        )
        return AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

    return transformers.PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )


def tokenize(
    batch: List[Dict[str, Any]],
    tokenizer: Union[PreTrainedTokenizerFast, str] = None,
    max_len: int = 2048,
) -> BatchEncoding:
    """
    Return the tokenized sequences given a batch of input data.
    Each data in the batch is expected to be a dictionary with "codons" and
    "organism" keys.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries with "codons" and
            "organism" keys.
        tokenizer (PreTrainedTokenizerFast, str, optional): The tokenizer object or
            path to the tokenizer file.
        max_len (int, optional): Maximum length of the tokenized sequence.

    Returns:
        BatchEncoding: The tokenized batch.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = load_tokenizer(tokenizer)

    tokenized = tokenizer(
        [data["codons"] for data in batch],
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Add token type IDs for species
    seq_len = tokenized["input_ids"].shape[-1]
    species_index = torch.tensor([[data["organism"]] for data in batch])
    tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

    return tokenized


def validate_and_convert_organism(organism: Union[int, str]) -> Tuple[int, str]:
    """
    Validate and convert the organism input to both ID and name.

    This function takes either an organism ID or name as input and returns both
    the ID and name. It performs validation to ensure the input corresponds to
    a valid organism in the ORGANISM2ID dictionary.

    Args:
        organism (Union[int, str]): Either the ID of the organism (int) or its
        name (str).

    Returns:
        Tuple[int, str]: A tuple containing the organism ID (int) and name (str).

    Raises:
        ValueError: If the input is neither a string nor an integer, if the
        organism name is not found in ORGANISM2ID, if the organism ID is not a
        value in ORGANISM2ID, or if no name is found for a given ID.

    Note:
        This function relies on the ORGANISM2ID dictionary imported from
        CodonTransformer.CodonUtils, which maps organism names to their
        corresponding IDs.
    """
    if isinstance(organism, str):
        if organism not in ORGANISM2ID:
            raise ValueError(
                f"Invalid organism name: {organism}. "
                "Please use a valid organism name or ID."
            )
        organism_id = ORGANISM2ID[organism]
        organism_name = organism

    elif isinstance(organism, int):
        if organism not in ORGANISM2ID.values():
            raise ValueError(
                f"Invalid organism ID: {organism}. "
                "Please use a valid organism name or ID."
            )

        organism_id = organism
        organism_name = next(
            (name for name, id in ORGANISM2ID.items() if id == organism), None
        )
        if organism_name is None:
            raise ValueError(f"No organism name found for ID: {organism}")

    return organism_id, organism_name


def get_high_frequency_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using High Frequency Choice (HFC) approach
    in which the most frequent codon for a given amino acid is always chosen.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select the most frequent codon for each amino acid in the protein sequence
    dna_codons = [
        codon_frequencies[aminoacid][0][np.argmax(codon_frequencies[aminoacid][1])]
        for aminoacid in protein
    ]
    return "".join(dna_codons)


def precompute_most_frequent_codons(
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
) -> Dict[str, str]:
    """
    Precompute the most frequent codon for each amino acid.

    Args:
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        Dict[str, str]: The most frequent codon for each amino acid.
    """
    # Create a dictionary mapping each amino acid to its most frequent codon
    return {
        aminoacid: codons[np.argmax(frequencies)]
        for aminoacid, (codons, frequencies) in codon_frequencies.items()
    }


def get_high_frequency_choice_sequence_optimized(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_high_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x10 faster speed.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Precompute the most frequent codons for each amino acid
    most_frequent_codons = precompute_most_frequent_codons(codon_frequencies)

    return "".join(most_frequent_codons[aminoacid] for aminoacid in protein)


def get_background_frequency_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using Background Frequency Choice (BFC)
    approach in which a random codon for a given amino acid is chosen using
    the codon frequencies probability distribution.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select a random codon for each amino acid based on the codon frequencies
    # probability distribution
    dna_codons = [
        np.random.choice(
            codon_frequencies[aminoacid][0], p=codon_frequencies[aminoacid][1]
        )
        for aminoacid in protein
    ]
    return "".join(dna_codons)


def precompute_cdf(
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
) -> Dict[str, Tuple[List[str], Any]]:
    """
    Precompute the cumulative distribution function (CDF) for each amino acid.

    Args:
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        Dict[str, Tuple[List[str], Any]]: CDFs for each amino acid.
    """
    cdf = {}

    # Calculate the cumulative distribution function for each amino acid
    for aminoacid, (codons, frequencies) in codon_frequencies.items():
        cdf[aminoacid] = (codons, np.cumsum(frequencies))

    return cdf


def get_background_frequency_choice_sequence_optimized(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_background_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x8 faster speed.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    dna_codons = []
    cdf = precompute_cdf(codon_frequencies)

    # Select a random codon for each amino acid using the precomputed CDFs
    for aminoacid in protein:
        codons, cumulative_prob = cdf[aminoacid]
        selected_codon_index = np.searchsorted(cumulative_prob, np.random.rand())
        dna_codons.append(codons[selected_codon_index])

    return "".join(dna_codons)


def get_uniform_random_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using Uniform Random Choice (URC) approach
    in which a random codon for a given amino acid is chosen using a uniform
    prior.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select a random codon for each amino acid using a uniform prior distribution
    dna_codons = []
    for aminoacid in protein:
        codons = codon_frequencies[aminoacid][0]
        random_index = np.random.randint(0, len(codons))
        dna_codons.append(codons[random_index])
    return "".join(dna_codons)


def get_icor_prediction(input_seq: str, model_path: str, stop_symbol: str) -> str:
    """
    Return the optimized codon sequence for the given protein sequence using ICOR.

    Credit: ICOR: improving codon optimization with recurrent neural networks
            Rishab Jain, Aditya Jain, Elizabeth Mauro, Kevin LeShane, Douglas
            Densmore

    Args:
        input_seq (str): The input protein sequence.
        model_path (str): The path to the ICOR model.
        stop_symbol (str): The symbol representing stop codons in the sequence.

    Returns:
        str: The optimized DNA sequence.
    """
    input_seq = input_seq.strip().upper()
    input_seq = input_seq.replace(stop_symbol, "*")

    # Define categorical labels from when model was trained.
    labels = [
        "AAA",
        "AAC",
        "AAG",
        "AAT",
        "ACA",
        "ACG",
        "ACT",
        "AGC",
        "ATA",
        "ATC",
        "ATG",
        "ATT",
        "CAA",
        "CAC",
        "CAG",
        "CCG",
        "CCT",
        "CTA",
        "CTC",
        "CTG",
        "CTT",
        "GAA",
        "GAT",
        "GCA",
        "GCC",
        "GCG",
        "GCT",
        "GGA",
        "GGC",
        "GTC",
        "GTG",
        "GTT",
        "TAA",
        "TAT",
        "TCA",
        "TCG",
        "TCT",
        "TGG",
        "TGT",
        "TTA",
        "TTC",
        "TTG",
        "TTT",
        "ACC",
        "CAT",
        "CCA",
        "CGG",
        "CGT",
        "GAC",
        "GAG",
        "GGT",
        "AGT",
        "GGG",
        "GTA",
        "TGC",
        "CCC",
        "CGA",
        "CGC",
        "TAC",
        "TAG",
        "TCC",
        "AGA",
        "AGG",
        "TGA",
    ]

    # Define aa to integer table
    def aa2int(seq: str) -> List[int]:
        _aa2int = {
            "A": 1,
            "R": 2,
            "N": 3,
            "D": 4,
            "C": 5,
            "Q": 6,
            "E": 7,
            "G": 8,
            "H": 9,
            "I": 10,
            "L": 11,
            "K": 12,
            "M": 13,
            "F": 14,
            "P": 15,
            "S": 16,
            "T": 17,
            "W": 18,
            "Y": 19,
            "V": 20,
            "B": 21,
            "Z": 22,
            "X": 23,
            "*": 24,
            "-": 25,
            "?": 26,
        }
        return [_aa2int[i] for i in seq]

    # Create empty array to fill
    oh_array = np.zeros(shape=(26, len(input_seq)))

    # Load placements from aa2int
    aa_placement = aa2int(input_seq)

    # One-hot encode the amino acid sequence:

    # style nit: more pythonic to write for i in range(0, len(aa_placement)):
    for i in range(0, len(aa_placement)):
        oh_array[aa_placement[i], i] = 1
        i += 1

    oh_array = [oh_array]
    x = np.array(np.transpose(oh_array))

    y = x.astype(np.float32)

    y = np.reshape(y, (y.shape[0], 1, 26))

    # Start ICOR session using model.
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    # Get prediction:
    pred_onx = sess.run(None, {input_name: y})

    # Get the index of the highest probability from softmax output:
    pred_indices = []
    for pred in pred_onx[0]:
        pred_indices.append(np.argmax(pred))

    out_str = ""
    for index in pred_indices:
        out_str += labels[index]

    return out_str


# ============================================================================
# CONSTRAINED BEAM SEARCH - "GUARDRAIL" SYSTEM
# ============================================================================
# Expert-recommended hard constraint approach that guarantees 45-55% GC content
# by enforcing the constraint DURING generation, not after.

def predict_dna_sequence_constrained_beam_search(
    protein: str,
    organism: Union[int, str],
    device: torch.device,
    tokenizer: Union[str, PreTrainedTokenizerFast] = None,
    model: Union[str, torch.nn.Module] = None,
    gc_target_min: float = 0.45,
    gc_target_max: float = 0.55,
    beam_size: Optional[int] = None,  # Will be adaptive based on sequence length
    match_protein: bool = True,
    temperature: float = 0.2,
    verbose: bool = False,
    gc_weight: float = 0.5,  # Weight for GC penalty in scoring
) -> DNASequencePrediction:
    """
    Generate DNA sequence with ENHANCED GC-aware constrained beam search.

    This is the improved "Guardrail" approach with smart fixes for long sequences:
    1. Adaptive beam size based on sequence length
    2. GC-aware scoring that penalizes deviations from 50% target
    3. Better lookahead with conservative GC budget calculations
    4. GC checkpoint tracking for course correction
    5. Restart mechanism when beams are exhausted

    Strategy:
    1. Maintain multiple candidate sequences (adaptive beam size)
    2. At each step, expand beams with GC-aware scoring
    3. Calculate conservative GC "budget" for each partial sequence
    4. PRUNE beams that cannot possibly reach target GC range
    5. Monitor GC at checkpoints and adjust course
    6. Restart with relaxed constraints if needed

    Args:
        protein: Input protein sequence
        organism: Target organism for optimization
        device: PyTorch device
        tokenizer: Model tokenizer (optional)
        model: Trained model (optional)
        gc_target_min: Minimum allowed GC content (default: 0.45)
        gc_target_max: Maximum allowed GC content (default: 0.55)
        beam_size: Number of candidate sequences (None for adaptive)
        match_protein: Only consider valid codons for each amino acid
        temperature: Sampling temperature
        verbose: Print debugging information
        gc_weight: Weight for GC penalty in scoring (default: 0.5)

    Returns:
        DNASequencePrediction with guaranteed GC content in target range
    """

    # Load tokenizer and model if needed
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        warnings.warn("Tokenizer path not provided. Loading from HuggingFace.", UserWarning)
    elif isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if model is None:
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        warnings.warn("Model path not provided. Loading from HuggingFace.", UserWarning)
    elif isinstance(model, str):
        model = load_model(model_path=model, device=device)

    model.eval()
    model.to(device)

    # Convert organism to ID if needed
    organism_id = ORGANISM2ID.get(organism, organism)
    if not isinstance(organism_id, int):
        raise ValueError(f"Invalid organism: {organism}")

    organism_name = next(name for name, id in ORGANISM2ID.items() if id == organism_id)

    # Adaptive beam size based on sequence length
    if beam_size is None:
        if len(protein) <= 50:
            beam_size = 20
        elif len(protein) <= 100:
            beam_size = 40
        elif len(protein) <= 150:
            beam_size = 60
        else:
            beam_size = 80  # Large beam for very long sequences

    # GC target center for scoring
    gc_target_center = (gc_target_min + gc_target_max) / 2.0

    if verbose:
        print(f"🎯 ENHANCED Constrained Beam Search (GC: {gc_target_min:.1%}-{gc_target_max:.1%})")
        print(f"   Protein: {protein[:40]}... ({len(protein)} aa)")
        print(f"   Adaptive beam size: {beam_size}")
        print(f"   GC target center: {gc_target_center:.1%}")

    # Pre-compute exact per-amino-acid GC bounds (Priority Fix #1)
    AA_MIN_GC, AA_MAX_GC = _compute_amino_acid_gc_bounds()

    # Create GC content lookup for codons
    codon_gc_content = _create_codon_gc_lookup()

    # Prepare input
    merged_seq = get_merged_seq(protein, None)
    input_dict = {
        "codons": merged_seq,
        "organism": organism_id,
    }
    tokenized_input = tokenize([input_dict], tokenizer=tokenizer).to(device)

    # Get model predictions
    with torch.no_grad():
        output_dict = model(**tokenized_input, return_dict=True)
        logits = output_dict.logits.detach().cpu()
        logits = logits[:, 1:-1, :]  # Remove [CLS] and [SEP] tokens

    # Prepare protein-specific codon constraints
    if match_protein:
        possible_tokens_per_position = [
            AMINO_ACID_TO_INDEX[token[0]] for token in merged_seq.split(" ")
        ]
        seq_len = logits.shape[1]
        if len(possible_tokens_per_position) > seq_len:
            possible_tokens_per_position = possible_tokens_per_position[:seq_len]
    else:
        possible_tokens_per_position = [list(range(logits.shape[-1]))] * logits.shape[1]

    # Initialize enhanced beam search
    # Each beam: (sequence_indices, sequence_dna, total_score, gc_content, gc_penalty_score)
    beams = [([], "", 0.0, 0.0, 0.0)]
    sequence_length = len(possible_tokens_per_position)
    checkpoint_interval = max(10, sequence_length // 10)  # GC checkpoints every 10% of sequence

    # Priority Fix #3: Adaptive beam rescue variables
    min_beam = max(8, beam_size // 10)  # Minimum beam size threshold
    relaxation_level = 0  # Track how much we've relaxed constraints
    saved_pruned_sequences = []  # Queue for rescued sequences
    rescue_interval = 10  # Check for rescue every 10 positions

    if verbose:
        print(f"   Sequence length: {sequence_length} codons")
        print(f"   GC checkpoints every {checkpoint_interval} positions")

    # Enhanced beam search with GC-aware scoring
    for pos in range(sequence_length):
        if verbose and pos % 10 == 0:
            print(f"   Position {pos}/{sequence_length}, beams: {len(beams)}")

        new_beams = []
        position_logits = logits[0, pos, :]  # Logits for this position
        possible_tokens = possible_tokens_per_position[pos]

        # Sort tokens by logits for better exploration
        token_scores = [(token_idx, position_logits[token_idx].item()) for token_idx in possible_tokens]
        token_scores.sort(key=lambda x: x[1], reverse=True)

        for seq_indices, seq_dna, score, current_gc, gc_penalty in beams:
            # Consider each possible token for this position
            for token_idx, token_logit in token_scores:
                token_name = INDEX2TOKEN[token_idx]
                codon = token_name[-3:].upper()  # Extract codon sequence

                # Calculate new sequence properties
                new_seq_indices = seq_indices + [token_idx]
                new_seq_dna = seq_dna + codon
                new_gc = _calculate_gc_content_fast(new_seq_dna)

                # ENHANCED GC-AWARE SCORING
                # Base score from model
                base_score = score + token_logit

                # SMART SCORING: Model likelihood first, GC as tie-breaker
                # Only apply penalty for significant GC deviations (>3%)
                gc_deviation = abs(new_gc - gc_target_center)

                if gc_deviation > 0.03:  # Only penalize if >3% deviation
                    progress = (pos + 1) / sequence_length  # 0 → 1
                    lambda_gc = 0.01 + 0.04 * progress  # Very gentle: 0.01 → 0.05
                    gc_penalty_component = lambda_gc * (gc_deviation - 0.03) * 5  # Only penalty above 3%
                    total_score = base_score - gc_penalty_component
                else:
                    # No penalty for small deviations - preserve model likelihood
                    gc_penalty_component = 0.0
                    total_score = base_score

                # EXACT GC CONSTRAINT CHECK (Priority Fix #1)
                remaining_positions = sequence_length - pos - 1
                if remaining_positions > 0:
                    # Calculate exact GC bounds based on remaining amino acids
                    remaining_aas = protein[pos + 1:]
                    min_possible_gc, max_possible_gc = _calculate_exact_gc_bounds(
                        new_seq_dna, remaining_aas, AA_MIN_GC, AA_MAX_GC
                    )

                    # PRUNE if impossible to reach target with exact bounds
                    if max_possible_gc < gc_target_min or min_possible_gc > gc_target_max:
                        # Save pruned sequence for potential rescue (Priority Fix #3)
                        if len(saved_pruned_sequences) < beam_size * 2:  # Limit queue size
                            saved_pruned_sequences.append((new_seq_indices, new_seq_dna, total_score, new_gc, gc_penalty_component))
                        continue  # Skip this beam - cannot reach target

                else:
                    # Final position - check if within target range
                    if not (gc_target_min <= new_gc <= gc_target_max):
                        # Save pruned sequence for potential rescue (Priority Fix #3)
                        if len(saved_pruned_sequences) < beam_size * 2:  # Limit queue size
                            saved_pruned_sequences.append((new_seq_indices, new_seq_dna, total_score, new_gc, gc_penalty_component))
                        continue  # Skip - final GC outside target

                # Add to candidate beams
                new_beams.append((new_seq_indices, new_seq_dna, total_score, new_gc, gc_penalty_component))

        # ENHANCED RESTART MECHANISM
        if len(new_beams) == 0:
            if verbose:
                print(f"   ❌ No valid beams at position {pos}! Activating restart mechanism...")

            # Restart with relaxed constraints or backtracking
            temp_gc_min = max(0.40, gc_target_min - 0.05)  # Relax by 5%
            temp_gc_max = min(0.60, gc_target_max + 0.05)

            if verbose:
                print(f"   🔄 Relaxing constraints to {temp_gc_min:.1%}-{temp_gc_max:.1%}")

            # Try with relaxed constraints
            for seq_indices, seq_dna, score, current_gc, gc_penalty in beams:
                for token_idx, token_logit in token_scores[:10]:  # Try top 10 tokens
                    token_name = INDEX2TOKEN[token_idx]
                    codon = token_name[-3:].upper()
                    new_seq_indices = seq_indices + [token_idx]
                    new_seq_dna = seq_dna + codon
                    new_gc = _calculate_gc_content_fast(new_seq_dna)

                    # Check with relaxed constraints
                    remaining_positions = sequence_length - pos - 1
                    if remaining_positions > 0:
                        remaining_aas = protein[pos + 1:]
                        min_possible_gc, max_possible_gc = _calculate_exact_gc_bounds(
                            new_seq_dna, remaining_aas, AA_MIN_GC, AA_MAX_GC
                        )
                        if max_possible_gc < temp_gc_min or min_possible_gc > temp_gc_max:
                            continue
                    else:
                        if not (temp_gc_min <= new_gc <= temp_gc_max):
                            continue

                    # GC-aware scoring with relaxed constraints - preserve model likelihood
                    base_score = score + token_logit
                    gc_deviation = abs(new_gc - gc_target_center)

                    if gc_deviation > 0.03:  # Only penalize if >3% deviation
                        progress = (pos + 1) / sequence_length  # 0 → 1
                        lambda_gc = 0.01 + 0.04 * progress  # Very gentle: 0.01 → 0.05
                        gc_penalty_component = lambda_gc * (gc_deviation - 0.03) * 5  # Only penalty above 3%
                        total_score = base_score - gc_penalty_component
                    else:
                        # No penalty for small deviations - preserve model likelihood
                        gc_penalty_component = 0.0
                        total_score = base_score

                    new_beams.append((new_seq_indices, new_seq_dna, total_score, new_gc, gc_penalty_component))
                    if len(new_beams) >= beam_size:
                        break
                if len(new_beams) >= beam_size:
                    break

        # Sort by GC-aware score and keep top beams
        new_beams.sort(key=lambda x: x[2], reverse=True)  # Sort by total score (with GC penalty)
        beams = new_beams[:beam_size]

        # ADAPTIVE BEAM RESCUE (Priority Fix #3)
        if len(beams) < min_beam and pos % rescue_interval == 0 and relaxation_level < 2:
            if verbose:
                print(f"   🚨 Beam rescue triggered: {len(beams)} < {min_beam} beams at position {pos}")

            relaxation_level += 1
            # Widen GC band by 1%
            temp_gc_min = max(0.40, gc_target_min - (relaxation_level * 0.01))
            temp_gc_max = min(0.60, gc_target_max + (relaxation_level * 0.01))

            if verbose:
                print(f"   🔄 Rescue level {relaxation_level}: GC band {temp_gc_min:.1%}-{temp_gc_max:.1%}")

            # Re-insert saved pruned sequences that now fit relaxed constraints
            rescue_candidates = []
            for seq_indices, seq_dna, score, gc_content, gc_penalty in saved_pruned_sequences:
                # Check if sequence fits relaxed constraints
                if temp_gc_min <= gc_content <= temp_gc_max:
                    # Soften penalty slightly for rescued sequences
                    softened_penalty = gc_penalty * (0.8 ** relaxation_level)
                    adjusted_score = score + (gc_penalty - softened_penalty)  # Recover some penalty
                    rescue_candidates.append((seq_indices, seq_dna, adjusted_score, gc_content, softened_penalty))

            # Add best rescue candidates to beams
            rescue_candidates.sort(key=lambda x: x[2], reverse=True)
            beams.extend(rescue_candidates[:min_beam - len(beams)])

            if verbose:
                print(f"   ✅ Rescued {len(rescue_candidates[:min_beam - len(beams)])} sequences, total beams: {len(beams)}")

        # GC CHECKPOINT MONITORING
        if pos % checkpoint_interval == 0 and pos > 0:
            if beams:
                avg_gc = sum(beam[3] for beam in beams) / len(beams)
                if verbose:
                    print(f"   📊 Checkpoint {pos}/{sequence_length}: Avg GC = {avg_gc:.1%}")

        if verbose and pos % 10 == 0 and beams:
            best_beam = beams[0]
            print(f"     Best beam GC: {best_beam[3]:.1%}, Score: {best_beam[2]:.2f}, GC penalty: {best_beam[4]:.1f}")

    # Select best final beam with SMART SELECTION
    if len(beams) == 0:
        raise RuntimeError("No valid sequences found with GC constraints!")

    # Sort beams by MODEL LIKELIHOOD first, then GC proximity as tie-breaker
    def beam_score(beam):
        seq_indices, seq_dna, total_score, gc_content, gc_penalty = beam
        # Primary: model likelihood (higher is better)
        # Secondary: GC proximity (closer to target is better)
        gc_proximity = abs(gc_content - gc_target_center)
        return (total_score, -gc_proximity)  # Model score first, then GC proximity

    beams.sort(key=beam_score)
    best_beam = beams[0]
    final_sequence = best_beam[1]
    final_gc = best_beam[3]
    final_score = best_beam[2]
    final_gc_penalty = best_beam[4]

    if verbose:
        print(f"   🎉 Final sequence: {final_sequence[:30]}...")
        print(f"   🎯 Final GC: {final_gc:.1%} (target: {gc_target_min:.1%}-{gc_target_max:.1%})")
        print(f"   📊 Final score: {final_score:.2f}, GC penalty: {final_gc_penalty:.1f}")
        print(f"   ✅ Constraint satisfied: {gc_target_min <= final_gc <= gc_target_max}")

    # Verify constraint compliance
    if not (gc_target_min <= final_gc <= gc_target_max):
        # One last attempt with the most GC-friendly beam
        for beam in beams:
            beam_gc = beam[3]
            if gc_target_min <= beam_gc <= gc_target_max:
                if verbose:
                    print(f"   🔄 Using alternative beam with GC: {beam_gc:.1%}")
                final_sequence = beam[1]
                final_gc = beam_gc
                break
        else:
            raise RuntimeError(f"CONSTRAINT VIOLATION: GC={final_gc:.1%} outside target range!")

    return DNASequencePrediction(
        organism=organism_name,
        protein=protein,
        processed_input=merged_seq,
        predicted_dna=final_sequence,
    )


def _create_codon_gc_lookup():
    """Create lookup table for GC content of each codon."""
    codon_gc = {}
    for token_idx, token_name in INDEX2TOKEN.items():
        if "_" in token_name:
            codon = token_name.split("_")[-1].upper()
            if len(codon) == 3:
                gc_count = codon.count('G') + codon.count('C')
                codon_gc[codon] = gc_count / 3.0
    return codon_gc


def _compute_amino_acid_gc_bounds():
    """Pre-compute exact per-amino-acid GC bounds (Priority Fix #1)."""
    AA_MIN_GC = {}
    AA_MAX_GC = {}

    for aa, token_indices in AMINO_ACID_TO_INDEX.items():
        gc_counts = []
        for token_idx in token_indices:
            token_name = INDEX2TOKEN[token_idx]
            if "_" in token_name:
                codon = token_name.split("_")[-1].upper()
                if len(codon) == 3:
                    gc_count = codon.count('G') + codon.count('C')
                    gc_counts.append(gc_count)

        if gc_counts:
            AA_MIN_GC[aa] = min(gc_counts)
            AA_MAX_GC[aa] = max(gc_counts)
        else:
            # Fallback for edge cases
            AA_MIN_GC[aa] = 0
            AA_MAX_GC[aa] = 3

    return AA_MIN_GC, AA_MAX_GC


def _calculate_exact_gc_bounds(current_seq_dna, remaining_aas, AA_MIN_GC, AA_MAX_GC):
    """Calculate exact GC bounds based on remaining amino acids (Priority Fix #1)."""
    current_gc_count = current_seq_dna.count('G') + current_seq_dna.count('C')

    # Calculate minimum and maximum possible GC from remaining amino acids
    rem_min = sum(AA_MIN_GC.get(aa, 0) for aa in remaining_aas)
    rem_max = sum(AA_MAX_GC.get(aa, 3) for aa in remaining_aas)

    # Calculate total sequence length
    total_len = len(current_seq_dna) + len(remaining_aas) * 3

    # Calculate bounds as percentages
    min_possible_gc = (current_gc_count + rem_min) / total_len
    max_possible_gc = (current_gc_count + rem_max) / total_len

    return min_possible_gc, max_possible_gc


def _calculate_gc_content_fast(dna_sequence: str) -> float:
    """Fast GC content calculation."""
    if not dna_sequence:
        return 0.0
    gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    return gc_count / len(dna_sequence)


def _calculate_min_possible_gc(partial_sequence: str, remaining_positions: int) -> float:
    """Calculate minimum possible GC content if remaining positions use all-AT codons."""
    if remaining_positions == 0:
        return _calculate_gc_content_fast(partial_sequence)

    # Add all-AT codons (0% GC) for remaining positions
    total_length = len(partial_sequence) + (remaining_positions * 3)
    current_gc_count = partial_sequence.count('G') + partial_sequence.count('C')
    # No additional GC from remaining AT codons
    return current_gc_count / total_length


def _calculate_max_possible_gc(partial_sequence: str, remaining_positions: int) -> float:
    """Calculate maximum possible GC content if remaining positions use all-GC codons."""
    if remaining_positions == 0:
        return _calculate_gc_content_fast(partial_sequence)

    # Add all-GC codons (100% GC) for remaining positions
    total_length = len(partial_sequence) + (remaining_positions * 3)
    current_gc_count = partial_sequence.count('G') + partial_sequence.count('C')
    additional_gc = remaining_positions * 3  # All remaining positions are GC
    return (current_gc_count + additional_gc) / total_length


def _calculate_min_possible_gc_conservative(partial_sequence: str, remaining_positions: int) -> float:
    """Conservative estimate of minimum possible GC content (DEPRECATED - use exact bounds)."""
    if remaining_positions == 0:
        return _calculate_gc_content_fast(partial_sequence)

    # Use slightly higher estimate than theoretical minimum for safety margin
    total_length = len(partial_sequence) + (remaining_positions * 3)
    current_gc_count = partial_sequence.count('G') + partial_sequence.count('C')

    # Conservative estimate: assume 10% GC for remaining positions (safety margin)
    conservative_additional_gc = remaining_positions * 3 * 0.1
    return (current_gc_count + conservative_additional_gc) / total_length


def _calculate_max_possible_gc_conservative(partial_sequence: str, remaining_positions: int) -> float:
    """Conservative estimate of maximum possible GC content (DEPRECATED - use exact bounds)."""
    if remaining_positions == 0:
        return _calculate_gc_content_fast(partial_sequence)

    # Use slightly lower estimate than theoretical maximum for safety margin
    total_length = len(partial_sequence) + (remaining_positions * 3)
    current_gc_count = partial_sequence.count('G') + partial_sequence.count('C')

    # Conservative estimate: assume 90% GC for remaining positions (safety margin)
    conservative_additional_gc = remaining_positions * 3 * 0.9
    return (current_gc_count + conservative_additional_gc) / total_length
