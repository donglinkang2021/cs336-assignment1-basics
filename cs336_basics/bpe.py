"""
A module for Byte-Pair Encoding (BPE) tokenizer training.
"""

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx.
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_bpe(text: str, vocab_size: int, special_tokens: list[str]):
    """
    Train a BPE tokenizer from a given text.
    """
    assert vocab_size >= 256

    num_merges = vocab_size - 256 - len(special_tokens)

    # For this basic implementation, we'll treat the whole text as one sequence.
    # A more advanced tokenizer might split by regex patterns.
    text_bytes = text.encode("utf-8")
    ids = [list(text_bytes)]

    # The vocabulary starts with all single bytes.
    vocab = {idx: bytes([idx]) for idx in range(256)}
    # The list of merges, which we will build up.
    merges_list = []

    for i in range(num_merges):
        # Count pairs in all our sequences.
        stats = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)

        # Find the most frequent pair.
        if not stats:
            break  # No more pairs to merge

        pair = max(stats, key=stats.get)

        # The new token id is the next available integer.
        idx = 256 + i

        # Merge the most frequent pair into a new token.
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

        # The merge is represented by the byte values of the merged tokens.
        merges_list.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    # Add special tokens to the vocabulary.
    # They get the highest token IDs.
    for i, token_str in enumerate(special_tokens):
        idx = 256 + num_merges + i
        vocab[idx] = token_str.encode("utf-8")

    return vocab, merges_list
