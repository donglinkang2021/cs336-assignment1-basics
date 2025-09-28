import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, get_context
from collections import defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)

# uv run pytest tests/test_train_bpe.py
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    task_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(bounds[:-1], bounds[1:])
    ]
    with get_context("forkserver").Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args) # list[list[list[int]]]
    
    # 3. Compute BPE merges
    ids: list[list[int]] = [
        token_ids for chunk_ids in chunk_results for token_ids in chunk_ids
    ] # chunk_ids is list[token_ids], token_ids is list[int]
    merges: list[tuple[int, int]] = []
    
    # Get all pairs from the pre-tokenized bytes
    pair_to_indices, counts = _get_pair_counts(ids)

    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not counts:
            break
        
        # Find the most frequent pair
        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return counts[pair], (vocab[pair[0]], vocab[pair[1]])
        max_pair = max(counts, key=rank)
        new_token = vocab[max_pair[0]] + vocab[max_pair[1]]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(max_pair)

        # Merge the most frequent pair in all affected tokens
        # Use affected_indices to only update tokens that contain the max_pair
        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token_ids = ids[j]
            if len(token_ids) < 2:
                continue
            # 1. Decrement counts for pairs in the old token
            # just ignore all the pair count in the old token right now
            for pair in zip(token_ids, token_ids[1:]):
                counts[pair] -= 1
                pair_to_indices[pair].discard(j)
                if counts[pair] == 0:
                    del counts[pair]
                    del pair_to_indices[pair]

            # 2. Merge the pair to create the new token
            new_token_ids = _merge_pair(token_ids, max_pair, new_id)

            # 3. Increment counts for pairs in the new token
            # add all the pair count in the new token
            for pair in zip(new_token_ids, new_token_ids[1:]):
                counts[pair] += 1
                pair_to_indices[pair].add(j)

            ids[j] = new_token_ids

    merges = [(vocab[a], vocab[b]) for a, b in merges]
    return vocab, merges

def _get_pair_counts(
        ids: list[list[int]]
    ) -> tuple[
        defaultdict[tuple[int, int], set], 
        defaultdict[tuple[int, int], int]
    ]:
    """Counts initial byte pair frequencies."""
    pair_to_indices = defaultdict(set)
    counts = defaultdict(int)
    for i, token_ids in enumerate(ids):
        for pair in zip(token_ids, token_ids[1:]):
            pair_to_indices[pair].add(i)
            counts[pair] += 1
    return pair_to_indices, counts

def _merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Merges a pair of bytes into a new token within a list of bytes."""
    new_token_ids = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
            new_token_ids.append(new_id)
            i += 2
        else:
            new_token_ids.append(token_ids[i])
            i += 1
    return new_token_ids

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), \
        "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = file_size
    mini = 4096  # 4k scan step (bigger save syscall)
    for bi in range(1, len(bounds) - 1):
        pos = bounds[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if not buf:
                bounds[bi] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                bounds[bi] = pos + found
                break
            pos += len(buf)
    return sorted(set(bounds))


def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    # 1. Remove special tokens by splitting the chunk at those tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)
    # 2. Pre-tokenize and count byte pair frequencies
    chunk_ids: list[list[int]] = []
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in _COMPILED_PAT.finditer(doc)]
        chunk_ids.extend([list(token) for token in tokens]) # list(bytes) -> list[int]
    return chunk_ids
