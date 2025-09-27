import regex as re

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

# uv run pytest tests/test_tokenizer.py
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initialize the Tokenizer with a vocabulary, BPE merge rules, and optional special tokens.

        Args:
            vocab (dict[int, bytes]): A mapping from token IDs to byte-encoded tokens.
            merges (list[tuple[bytes, bytes]]): A list of merge operations as tuples of byte pairs.
            special_tokens (list[str] | None): Optional list of user-defined special tokens to include.
        """
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        self.merges = merges
        # Create a dictionary for quick lookup of merge ranks
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath (str): Path to the vocabulary file (from BPE training).
            merges_filepath (str): Path to the merges file (from BPE training).
            special_tokens (list[str] | None): Optional list of special tokens to include.

        Returns:
            Tokenizer: A Tokenizer instance initialized with the given files.
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = eval(token_str).encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((eval(parts[0]).encode("utf-8"), eval(parts[1]).encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs using the BPE algorithm.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        token_ids = []
        # 1. Pre-tokenize the input text into chunks, handling special tokens.
        pre_tokens_list = process_chunk((text, self.special_tokens, True))

        for tokens in pre_tokens_list:
            # If the chunk is a special token, directly look up its ID.
            if len(tokens) == 1 and tokens[0] in self.vocab_reversed:
                token_id = self.vocab_reversed.get(tokens[0])
                if token_id is not None:
                    token_ids.append(token_id)
                    continue

            # 2. Iteratively merge pairs in each chunk based on merge ranks.
            while len(tokens) >= 2:
                # Find all current pairs in the token list.
                pairs = list(zip(tokens, tokens[1:]))
                
                # Find the pair with the lowest merge rank (highest priority).
                # Use a large number (infinity) for pairs not in our merge rules.
                best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

                # If the best pair is not in our merge rules, no more merges are possible.
                if best_pair not in self.merge_ranks:
                    break
                
                # 3. Merge the best pair into a new token.
                new_tok = best_pair[0] + best_pair[1]
                tokens = _merge_pair(tokens, best_pair, new_tok)
            
            # 4. Convert the final merged tokens to their corresponding IDs.
            for token in tokens:
                token_id = self.vocab_reversed.get(token)
                if token_id is not None:
                    token_ids.append(token_id)
        
        return token_ids


    def encode_iterable(self, iterable: list[str]) -> iter:
        """
        Lazily encode an iterable of strings into a stream of token IDs.

        Useful for memory-efficient tokenization of large datasets.

        Args:
            iterable (list[str]): An iterable of strings (e.g., lines from a file).

        Returns:
            iter: A generator that yields token IDs one at a time.
        """
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = b"".join(self.vocab.get(token_id, b'\xef\xbf\xbd') for token_id in ids)
        return tokens.decode(encoding='utf-8', errors='replace')
