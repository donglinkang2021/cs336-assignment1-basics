from collections import defaultdict
import regex as re

# Regex pattern from GPT-2, see 2.4 BPE Tokenizer Training.md
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

special_tokens = ["<|endoftext|>", "<|pad|>"]

file_path = "tests/fixtures/tinystories_sample.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

word_freqs = _process_chunk(text, special_tokens)
print(f"Processed {len(word_freqs)} unique words/tokens.")
print(list(word_freqs.items()))  # Print first 10 items for inspection