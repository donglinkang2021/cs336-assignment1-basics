import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, get_context
from collections import defaultdict

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
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    task_args = [(chunk, special_tokens, False) for chunk in chunk_list]
    with get_context("forkserver").Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 3. Compute BPE merges
    pre_tokens_bytes: list[list[bytes]] = [
        token for chunk in chunk_results for token in chunk
    ] # list[token], each token is list[bytes]
    merges: list[tuple[bytes, bytes]] = []
    
    # Get all pairs from the pre-tokenized bytes
    pair_to_indices, counts = _get_pair_counts(pre_tokens_bytes)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break
        
        # Find the most frequent pair
        max_pair = max(counts, key=lambda p: (counts[p], p))

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        # Merge the most frequent pair in all affected tokens
        # Use affected_indices to only update tokens that contain the max_pair
        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token_bytes = pre_tokens_bytes[j]
            if len(token_bytes) < 2:
                continue
            # 1. Decrement counts for pairs in the old token
            # just ignore all the pair count in the old token right now
            _update_counts(j, token_bytes, pair_to_indices, counts, remove=True)

            # 2. Merge the pair to create the new token
            # token_bytes[b'a', b'b', b'c'], max_pair(b'a', b'b') 
            # -> new_token_bytes[b'ab', b'c']
            new_token_bytes = _merge_pair(token_bytes, max_pair, new_token)

            # 3. Increment counts for pairs in the new token
            # add all the pair count in the new token
            _update_counts(j, new_token_bytes, pair_to_indices, counts, remove=False)

            pre_tokens_bytes[j] = new_token_bytes

    return vocab, merges

def _get_pair_counts(
        pre_tokens_bytes: list[list[bytes]]
    ) -> tuple[
        defaultdict[tuple[bytes, bytes], set], 
        defaultdict[tuple[bytes, bytes], int]
    ]:
    """Counts initial byte pair frequencies."""
    pair_to_indices = defaultdict(set)
    counts = defaultdict(int)
    for i, token_bytes in enumerate(pre_tokens_bytes):
        for pair in zip(token_bytes, token_bytes[1:]):
            pair_to_indices[pair].add(i)
            counts[pair] += 1
    return pair_to_indices, counts

def _merge_pair(
    token_bytes: list[bytes], pair: tuple[bytes, bytes], new_token: bytes
) -> list[bytes]:
    """Merges a pair of bytes into a new token within a list of bytes."""
    new_token_bytes = []
    i = 0
    while i < len(token_bytes):
        if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == pair:
            new_token_bytes.append(new_token)
            i += 2
        else:
            new_token_bytes.append(token_bytes[i])
            i += 1
    return new_token_bytes

def _update_counts(
    token_idx: int,
    token_bytes: list[bytes],
    pair_to_indices: defaultdict[tuple[bytes, bytes], set],
    counts: defaultdict[tuple[bytes, bytes], int],
    *,
    remove: bool,
):
    """Updates pair counts and indices for a token."""
    for pair in zip(token_bytes, token_bytes[1:]):
        if remove:
            counts[pair] -= 1
            pair_to_indices[pair].discard(token_idx)
            if counts[pair] == 0:
                del counts[pair]
                del pair_to_indices[pair]
        else:
            counts[pair] += 1
            pair_to_indices[pair].add(token_idx)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    chunk, special_tokens, keep_special_tokens = args
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"
    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        if keep_special_tokens and segment in special_tokens:
            token_bytes = [segment.encode("utf-8")]
            pre_tokens_bytes.append(token_bytes)
        else:
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)
    return pre_tokens_bytes
