import regex as re
from .bpe import process_chunk, _merge_pair

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
        pre_tokens_list = process_chunk((text, self.special_tokens, True))
        for tokens in pre_tokens_list:
            for pair in self.merges:
                new_tok = pair[0] + pair[1]
                tokens = _merge_pair(tokens, pair, new_tok)
            
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
