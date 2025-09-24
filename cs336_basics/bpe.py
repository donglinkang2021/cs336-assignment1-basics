"""
A module for Byte-Pair Encoding (BPE) tokenizer training.
"""
import regex as re
from collections import defaultdict

# Regex pattern from GPT-2, see 2.4 BPE Tokenizer Training.md
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_stats(splits: dict[tuple[int, ...], int]) -> defaultdict[tuple[int, int], int]:
    """
    Given a dictionary of token sequences and their frequencies,
    return a dictionary of counts of consecutive pairs.
    """
    pair_counts = defaultdict(int)
    for sequence, freq in splits.items():
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])
            pair_counts[pair] += freq
    return pair_counts


def merge(sequence: tuple[int, ...], pair: tuple[int, int], idx: int) -> tuple[int, ...]:
    """
    In the sequence of integers, replace all consecutive occurrences
    of pair with the new integer token idx.
    """
    new_sequence = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i+1] == pair[1]:
            new_sequence.append(idx)
            i += 2
        else:
            new_sequence.append(sequence[i])
            i += 1
    return tuple(new_sequence)

def _process_chunk(text_chunk: str, special_tokens: list[str]) -> defaultdict[bytes, int]:
    """
    Processes a chunk of text: splits by special tokens, pre-tokenizes,
    and counts frequencies of resulting words.
    This function is designed to be called by parallel workers.
    """
    word_freqs = defaultdict(int)

    # Create a regex pattern to split the text by special tokens.
    # Escape special characters in tokens that might be regex metacharacters.
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        sub_chunks = re.split(f"({special_pattern})", text_chunk)
    else:
        sub_chunks = [text_chunk]

    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue
        # If the sub_chunk is a special token, we don't pre-tokenize it.
        # We count it as a whole word. The main training function will handle it.
        if sub_chunk in special_tokens:
            word_freqs[sub_chunk.encode("utf-8")] += 1
        else:
            # Apply the GPT-2 regex pre-tokenizer.
            for match in re.finditer(GPT2_PAT, sub_chunk):
                word_bytes = match.group(0).encode("utf-8")
                word_freqs[word_bytes] += 1
    return word_freqs


def train_bpe(text: str, vocab_size: int, special_tokens: list[str]):
    """
    Train a BPE tokenizer from a given text using regex pre-tokenization
    and an optimized merge loop.
    """
    assert vocab_size >= 256
    
    # 1. Pre-tokenize and count word frequencies.
    # This part can be parallelized by splitting `text` into chunks
    # and running _process_chunk on each chunk.
    word_freqs = _process_chunk(text, special_tokens)

    # Convert byte strings to tuples of integers.
    splits = {tuple(word): freq for word, freq in word_freqs.items()}

    # Base vocabulary consists of all 256 bytes.
    vocab = {idx: bytes([idx]) for idx in range(256)}
    
    # Add special tokens to the vocabulary first.
    # They are not part of the merging process.
    for i, token_str in enumerate(special_tokens):
        idx = 256 + i
        vocab[idx] = token_str.encode("utf-8")

    # The number of merges is the desired vocab size minus the initial tokens.
    num_merges = vocab_size - len(vocab)
    merges = {}  # (b1, b2) -> new_idx

    # 2. Iteratively merge the most frequent pair.
    for i in range(num_merges):
        # 3. Count pairs in all our pre-token sequences.
        stats = get_stats(splits)
        if not stats:
            break  # No more pairs to merge.

        # 4. Find the most frequent pair.
        # Deterministically break ties by preferring the lexicographically greater pair.
        pair = max(stats, key=lambda p: (stats[p], p))

        # The new token id is the next available integer.
        idx = 256 + len(special_tokens) + i

        # 5. Merge the most frequent pair.
        new_splits = {}
        for sequence, freq in splits.items():
            # This is still a full pass over the splits. A more advanced optimization
            # would be to only update the affected splits.
            new_sequence = merge(sequence, pair, idx)
            new_splits[new_sequence] = new_splits.get(new_sequence, 0) + freq
        splits = new_splits

        # Store the merge rule and update the vocabulary.
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    # Convert merges dict to the required list of tuples format.
    merges_list = sorted(merges.items(), key=lambda p: p[1])
    # The final list should contain byte values, not integer IDs.
    final_merges = []
    for (p0, p1), idx in merges_list:
        # We need to look up the byte representation for p0 and p1
        # which could themselves be merged tokens.
        final_merges.append((vocab[p0], vocab[p1]))

    return vocab, final_merges
