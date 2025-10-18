import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

def count_token_id_in_bin(
    bin_path: Path,
    token_id: int,
    dtype=np.uint16,
    chunk_size: int = 1024 * 1024  # 1M tokens per chunk
):
    """
    Count the occurrences of a specific token ID in a binary file.
    """
    if not bin_path.exists():
        print(f"Error: File '{bin_path}' does not exist")
        return None, 0

    file_size = bin_path.stat().st_size
    bytes_per_token = np.dtype(dtype).itemsize
    total_tokens = file_size // bytes_per_token

    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Target token ID: {token_id}")
    print("Start counting...")

    # Initialize counter
    total_count = 0
    
    with open(bin_path, 'rb') as f:
        with tqdm(total=total_tokens, desc="Counting progress", unit="tokens") as pbar:
            while True:
                # Read a chunk of data
                buffer = f.read(chunk_size * bytes_per_token)
                if not buffer:
                    break
                
                chunk = np.frombuffer(buffer, dtype=dtype)
                
                # Count occurrences in the current chunk
                total_count += np.count_nonzero(chunk == token_id)
                
                pbar.update(len(chunk))

    return total_count, total_tokens

def main():
    parser = argparse.ArgumentParser(description='Count the occurrences of a specific token ID in a binary file.')
    parser.add_argument('--bin_path', type=str, default='data/openwebtext-32k/train.bin',
                       help='Path to the binary file containing token IDs.')
    parser.add_argument('--token_id', type=int, required=True,
                       help='The token ID to count.')
    parser.add_argument('--chunk_size', type=int, default=1024*1024,
                       help='Number of tokens to process per chunk (default: 1M).')
    
    args = parser.parse_args()
    
    bin_path = Path(args.bin_path)
    
    print("=" * 60)
    print("Token ID Occurrence Counting Tool")
    print("=" * 60)
    
    start_time = time.time()
    
    count, total_tokens = count_token_id_in_bin(
        bin_path, 
        args.token_id,
        chunk_size=args.chunk_size
    )
    
    end_time = time.time()
    
    if count is not None:
        print("\nCounting results:")
        print("-" * 40)
        print(f"Token ID '{args.token_id}' appeared {count:,} times.")
        print(f"\nTotal tokens in file: {total_tokens:,}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print("=" * 60)

if __name__ == "__main__":
    main()