import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse
import time

def find_subsequence_occurrences(main_array, sub_array):
    """
    Efficiently find the number of occurrences of a subsequence in a numpy array.
    """
    if len(sub_array) == 0:
        return 0
    if len(sub_array) == 1:
        return np.count_nonzero(main_array == sub_array[0])
    
    # Create a sliding window view of the main array
    shape = main_array.shape[:-1] + (main_array.shape[-1] - len(sub_array) + 1, len(sub_array))
    strides = main_array.strides + (main_array.strides[-1],)
    windows = np.lib.stride_tricks.as_strided(main_array, shape=shape, strides=strides)
    
    # Compare windows with the subsequence and count matches
    return np.count_nonzero(np.all(windows == sub_array, axis=1))

def count_token_sequences_in_bin(
    bin_path: Path,
    target_sequences: dict, # e.g., {"'s": [27, 82]}
    dtype=np.uint16,
    chunk_size: int = 1024 * 1024  # 1M tokens per chunk
):
    """
    Count the occurrences of specific token sequences in a binary file.
    """
    if not bin_path.exists():
        print(f"Error: File '{bin_path}' does not exist")
        return None, 0

    file_size = bin_path.stat().st_size
    bytes_per_token = np.dtype(dtype).itemsize
    total_tokens = file_size // bytes_per_token

    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Total tokens: {total_tokens:,}")
    print("Target token sequences:")
    for string, seq in target_sequences.items():
        print(f"  '{string}': {seq}")
    print("Start counting...")

    # Initialize counters
    counts = {string: 0 for string in target_sequences.keys()}
    
    # Calculate the maximum length needed for overlap between chunks
    max_seq_len = 0
    if target_sequences:
        max_seq_len = max(len(seq) for seq in target_sequences.values())

    overlap_size = max(0, max_seq_len - 1)
    overlap = np.array([], dtype=dtype)

    with open(bin_path, 'rb') as f:
        with tqdm(total=total_tokens, desc="Counting progress", unit="tokens") as pbar:
            while True:
                # Read a chunk of data
                buffer = f.read(chunk_size * bytes_per_token)
                if not buffer:
                    break
                
                chunk = np.frombuffer(buffer, dtype=dtype)
                
                # Concatenate the overlap from the previous chunk with the current chunk
                data_to_search = np.concatenate((overlap, chunk))

                # Count each target sequence
                for string, seq_list in target_sequences.items():
                    seq = np.array(seq_list, dtype=dtype)
                    if len(seq) > 0 and len(data_to_search) >= len(seq):
                        counts[string] += find_subsequence_occurrences(data_to_search, seq)

                # Update the overlap for the next chunk
                if overlap_size > 0:
                    overlap = chunk[-overlap_size:]
                
                pbar.update(len(chunk))

    return counts, total_tokens

def get_token_sequences_from_tokenizer(tokenizer_path: Path, target_strings: list):
    """
    Get the token ID sequences for target strings from the tokenizer.
    """
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    token_sequences = {}
    for string in target_strings:
        # Encode a single string, add_special_tokens=False ensures we only get the tokens for the string itself
        encoding = tokenizer.encode(string, add_special_tokens=False)
        if encoding.ids:
            token_sequences[string] = encoding.ids
            print(f"'{string}' -> token_ids: {encoding.ids}")
        else:
            print(f"Warning: '{string}' cannot be encoded")
            token_sequences[string] = []
            
    return token_sequences

def main():
    parser = argparse.ArgumentParser(description='Count the occurrences of specific token sequences in a binary file')
    parser.add_argument('--bin_path', type=str, default='data/openwebtext-32k/train.bin',
                       help='Path to the binary file')
    parser.add_argument('--tokenizer_path', type=str, default='hf_tokenizer/openwebtext-32k/tokenizer.json',
                       help='Path to the tokenizer file')
    parser.add_argument('--chunk_size', type=int, default=1024*1024,
                       help='Number of tokens to process per chunk (1M)')
    
    args = parser.parse_args()
    
    bin_path = Path(args.bin_path)
    tokenizer_path = Path(args.tokenizer_path)
    
    # Target strings
    target_strings = ["鈥檚", "'s", "’s", "‘s"]
    
    print("=" * 60)
    print("Token Sequence Occurrence Counting Tool")
    print("=" * 60)
    
    # Get token sequences
    string_to_sequence = get_token_sequences_from_tokenizer(tokenizer_path, target_strings)
    
    print("-" * 60)
    
    # Count token sequence occurrences
    start_time = time.time()
    
    results, total_tokens = count_token_sequences_in_bin(
        bin_path, 
        string_to_sequence, 
        chunk_size=args.chunk_size
    )
    
    end_time = time.time()
    
    if results:
        print("\nCounting results:")
        print("-" * 40)
        
        for string, count in results.items():
            print(f"'{string}' (sequence: {string_to_sequence[string]}): {count:,} times")
        
        print(f"\nTotal tokens: {total_tokens:,}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print("=" * 60)

if __name__ == "__main__":
    main()