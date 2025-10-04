import time
import numpy as np
from pathlib import Path
import json
import random
from typing import List, Tuple
from tokenizers import Tokenizer

def load_sample_documents(dataset_name: str, n_samples: int = 10) -> List[str]:
    """Load sample documents from dataset"""
    documents = []
    
    if dataset_name == "tinystories":
        file_path = "data/TinyStoriesV2-GPT4-train.txt"
    elif dataset_name == "openwebtext":
        file_path = "data/owt_train.txt"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loading documents from {file_path}...")
    
    # Read file in chunks to avoid memory issues
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first 10MB to get samples
        chunk_size = 10 * 1024 * 1024  # 10MB
        content = f.read(chunk_size)
        
        # Split by <|endoftext|> and take random samples
        docs = content.split('<|endoftext|>')
        docs = [doc.strip() for doc in docs if doc.strip() and len(doc) > 50]  # Filter very short docs
        
        print(f"Found {len(docs)} documents in first {chunk_size//1024//1024}MB")
        
        if len(docs) > n_samples:
            documents = random.sample(docs, n_samples)
        else:
            documents = docs[:n_samples]
    
    print(f"Selected {len(documents)} documents")
    return documents

def calculate_compression_ratio(text: str, token_ids: List[int]) -> float:
    """Calculate compression ratio as bytes per token"""
    text_bytes = len(text.encode('utf-8'))
    return text_bytes / len(token_ids)

def experiment_compression_ratios():
    """Experiment (a): Calculate compression ratios"""
    print("=== Experiment (a): Compression Ratios ===")
    
    # Load tokenizers
    ts_tokenizer = Tokenizer.from_file("hf_tokenizer/tinystories/tokenizer.json")
    owt_tokenizer = Tokenizer.from_file("hf_tokenizer/openwebtext-32k/tokenizer.json")
    
    # Sample documents
    ts_docs = load_sample_documents("tinystories", 10)
    owt_docs = load_sample_documents("openwebtext", 10)
    
    # Test TinyStories tokenizer on TinyStories data
    ts_ratios = []
    for doc in ts_docs:
        encoded = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, encoded.ids)
        ts_ratios.append(ratio)
    
    # Test OpenWebText tokenizer on OpenWebText data
    owt_ratios = []
    for doc in owt_docs:
        encoded = owt_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, encoded.ids)
        owt_ratios.append(ratio)
    
    ts_avg = np.mean(ts_ratios)
    owt_avg = np.mean(owt_ratios)
    
    print(f"TinyStories tokenizer on TinyStories data: {ts_avg:.2f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText data: {owt_avg:.2f} bytes/token")
    
    return ts_avg, owt_avg, ts_docs, owt_docs

def experiment_cross_domain(ts_docs, owt_docs):
    """Experiment (b): Cross-domain tokenization"""
    print("\n=== Experiment (b): Cross-domain Tokenization ===")
    
    ts_tokenizer = Tokenizer.from_file("hf_tokenizer/tinystories/tokenizer.json")
    owt_tokenizer = Tokenizer.from_file("hf_tokenizer/openwebtext-32k/tokenizer.json")
    
    # Use TinyStories tokenizer on OpenWebText data
    cross_ratios = []
    for doc in owt_docs:
        encoded = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, encoded.ids)
        cross_ratios.append(ratio)
    
    # Compare with native OpenWebText tokenizer
    native_ratios = []
    for doc in owt_docs:
        encoded = owt_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, encoded.ids)
        native_ratios.append(ratio)
    
    cross_avg = np.mean(cross_ratios)
    native_avg = np.mean(native_ratios)
    
    print(f"TinyStories tokenizer on OpenWebText: {cross_avg:.2f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText: {native_avg:.2f} bytes/token")
    degradation = ((cross_avg - native_avg) / native_avg * 100)
    if degradation > 0:
        print(f"Degradation: {degradation:.1f}%")
    else:
        print(f"Improvement: {-degradation:.1f}% (unexpected - may indicate sampling bias)")
    
    return cross_avg, native_avg

def experiment_throughput():
    """Experiment (c): Throughput measurement"""
    print("\n=== Experiment (c): Throughput Measurement ===")
    
    tokenizer = Tokenizer.from_file("hf_tokenizer/openwebtext-32k/tokenizer.json")
    
    # Create test text
    test_text = "This is a test sentence for measuring tokenization throughput. " * 1000
    test_bytes = len(test_text.encode('utf-8'))
    
    # Measure tokenization time
    start_time = time.time()
    for _ in range(100):  # Multiple runs for better measurement
        encoded = tokenizer.encode(test_text)
    end_time = time.time()
    
    total_time = end_time - start_time
    total_bytes = test_bytes * 100
    throughput = total_bytes / total_time  # bytes per second
    
    # Estimate time for Pile dataset (825 GB)
    pile_bytes = 825 * 1024 * 1024 * 1024  # 825 GB in bytes
    pile_time_seconds = pile_bytes / throughput
    pile_time_hours = pile_time_seconds / 3600
    
    print(f"Tokenizer throughput: {throughput/1024/1024:.2f} MB/s")
    print(f"Estimated time for Pile dataset (825 GB): {pile_time_hours:.1f} hours")
    
    return throughput, pile_time_hours

def experiment_uint16_choice():
    """Experiment (d): Why uint16 is appropriate"""
    print("\n=== Experiment (d): uint16 Choice ===")
    
    # Check vocabulary sizes
    ts_tokenizer = Tokenizer.from_file("hf_tokenizer/tinystories/tokenizer.json")
    owt_tokenizer = Tokenizer.from_file("hf_tokenizer/openwebtext-32k/tokenizer.json")
    
    ts_vocab_size = ts_tokenizer.get_vocab_size()
    owt_vocab_size = owt_tokenizer.get_vocab_size()
    
    print(f"TinyStories vocabulary size: {ts_vocab_size}")
    print(f"OpenWebText vocabulary size: {owt_vocab_size}")
    print(f"uint16 max value: {2**16 - 1}")
    print(f"uint8 max value: {2**8 - 1}")
    print(f"uint32 max value: {2**32 - 1}")
    
    # Memory usage comparison
    sample_tokens = 1000000  # 1M tokens
    uint8_bytes = sample_tokens * 1
    uint16_bytes = sample_tokens * 2
    uint32_bytes = sample_tokens * 4
    
    print(f"\nMemory usage for 1M tokens:")
    print(f"uint8:  {uint8_bytes/1024/1024:.1f} MB")
    print(f"uint16: {uint16_bytes/1024/1024:.1f} MB")
    print(f"uint32: {uint32_bytes/1024/1024:.1f} MB")

def main():
    """Run all experiments"""
    random.seed(42)  # For reproducibility
    
    # Experiment (a)
    ts_avg, owt_avg, ts_docs, owt_docs = experiment_compression_ratios()
    
    # Experiment (b)
    cross_avg, native_avg = experiment_cross_domain(ts_docs, owt_docs)
    
    # Experiment (c)
    throughput, pile_time = experiment_throughput()
    
    # Experiment (d)
    experiment_uint16_choice()
    
    # Summary for LaTeX answers
    print("\n" + "="*50)
    print("SUMMARY FOR LATEX ANSWERS:")
    print("="*50)
    
    print(f"\n(a) TinyStories tokenizer achieves {ts_avg:.2f} bytes/token compression ratio, while OpenWebText tokenizer achieves {owt_avg:.2f} bytes/token, with the larger vocabulary size providing better compression efficiency.")
    degradation = ((native_avg - cross_avg) / native_avg * 100)
    print(f"\n(b) Using TinyStories tokenizer on OpenWebText data results in {cross_avg:.2f} bytes/token compared to {native_avg:.2f} bytes/token with the native tokenizer, showing a {degradation:.1f}% degradation due to vocabulary mismatch and increased unknown token fragmentation.")
    print(f"\n(c) The tokenizer achieves approximately {throughput/1024/1024:.2f} MB/s throughput, suggesting it would take around {pile_time:.1f} hours to tokenize the entire Pile dataset (825 GB).")    
    print(f"\n(d) uint16 is appropriate because it can represent values up to 65,535, which easily accommodates our vocabulary sizes (10K and 32K), while being more memory-efficient than uint32 and avoiding the limitations of uint8 (max 255).")

if __name__ == "__main__":
    main()