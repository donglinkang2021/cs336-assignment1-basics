import json
from collections import Counter

def compare_tokenizers(tinystories_path, owt_path):
    # Load vocabularies
    with open(f"{tinystories_path}/vocab.json", 'r') as f:
        ts_vocab = json.load(f)
    
    with open(f"{owt_path}/vocab.json", 'r') as f:
        owt_vocab = json.load(f)
    
    # 1. Vocabulary size comparison
    print(f"TinyStories vocab size: {len(ts_vocab)}")
    print(f"OpenWebText vocab size: {len(owt_vocab)}")
    
    # 2. Token length distribution
    ts_lengths = [len(token) for token in ts_vocab.keys()]
    owt_lengths = [len(token) for token in owt_vocab.keys()]
    
    print(f"TinyStories avg token length: {sum(ts_lengths)/len(ts_lengths):.2f}")
    print(f"OpenWebText avg token length: {sum(owt_lengths)/len(owt_lengths):.2f}")
    
    # 3. Character type analysis
    def analyze_chars(vocab):
        char_types = Counter()
        for token in vocab.keys():
            for char in token:
                if char.isalpha():
                    char_types['alphabetic'] += 1
                elif char.isdigit():
                    char_types['numeric'] += 1
                elif char.isspace():
                    char_types['whitespace'] += 1
                else:
                    char_types['special'] += 1
        return char_types
    
    ts_chars = analyze_chars(ts_vocab)
    owt_chars = analyze_chars(owt_vocab)
    
    print("Character distribution:")
    print(f"TinyStories: {dict(ts_chars)}")
    print(f"OpenWebText: {dict(owt_chars)}")
    
    # 4. Common tokens
    common_tokens = set(ts_vocab.keys()) & set(owt_vocab.keys())
    print(f"Common tokens: {len(common_tokens)}")
    
    # 5. Unique patterns
    ts_unique = set(ts_vocab.keys()) - set(owt_vocab.keys())
    owt_unique = set(owt_vocab.keys()) - set(ts_vocab.keys())
    
    print(f"TinyStories unique tokens: {len(ts_unique)}")
    print(f"OpenWebText unique tokens: {len(owt_unique)}")
    
    # Show some examples
    print("\nSample unique TinyStories tokens:")
    print(list(ts_unique)[:10])
    
    print("\nSample unique OpenWebText tokens:")
    print(list(owt_unique)[:10])

# Usage
compare_tokenizers("hf_tokenizer/tinystories", "hf_tokenizer/openwebtext-32k")