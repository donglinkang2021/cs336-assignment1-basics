import json
import time
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers import decoders

def main():
    """
    Trains a BPE tokenizer using the Hugging Face tokenizers library
    on the TinyStories dataset and saves the results for comparison.
    """
    # --- Configuration (mirrors train_bpe.py) ---
    # !!!IMPORTANT!!! 
    # Please replace this path with the actual path to your TinyStories dataset file
    file_list = [
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/TinyStoriesV2-GPT4-valid.txt",
        # "data/owt_train.txt",
        # "data/owt_valid.txt",
    ]
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    out_dir = "hf_tokenizer/tinystories"
    # out_dir = "hf_tokenizer/openwebtext"
    output_dir = Path(out_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "tokenizer.json"
    vocab_file = output_dir / "vocab.json"
    merges_file = output_dir / "merges.txt"

    # --- Tokenizer Initialization ---
    # Initialize a BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    

    # --- Training ---
    print("Starting BPE training with 'tokenizers' library...")
    
    # Configure the trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    start_time = time.time()
    
    # Train the tokenizer
    tokenizer.train(file_list, trainer)
    
    end_time = time.time()
    
    training_duration_seconds = end_time - start_time
    training_duration_minutes = training_duration_seconds / 60

    print(f"Training complete in {training_duration_seconds:.2f} seconds (~{training_duration_minutes:.2f} minutes).")
    
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # --- Smoke Test: Encode / Decode ---
    print("\n--- Smoke Test ---")
    sample_text = "Hello world! This is a test.\nNew line here."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

    print(f"Original text: '{sample_text}'")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Encoded tokens: {encoded.tokens}")
    print(f"Decoded text: '{decoded}'")

    if sample_text == decoded:
        print("✅ Encode/Decode round-trip successful!")
    else:
        print("❌ Mismatch in Encode/Decode round-trip.")
        print(f"   Original: '{sample_text}'")
        print(f"   Decoded:  '{decoded}'")

    # --- Save Results ---
    print(f"Saving results to '{output_dir}'...")

    # Save the complete tokenizer model (recommended way)
    tokenizer.save(str(model_file))

    # Save vocab and merges in the same format as your other script for comparison
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Extract and save merge rules
    # The BPE model state which includes merges is stored inside the saved JSON model.
    with open(model_file, "r", encoding="utf-8") as f:
        model_json = json.load(f)
    
    merges = model_json.get("model", {}).get("merges", [])
    
    with open(merges_file, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge}\n")

    print("Results saved.")

    # --- Analysis ---
    # Find the longest token
    longest_token = ""
    if vocab:
        longest_token = max(vocab.keys(), key=len)

    print("\n--- Hugging Face Tokenizer Experiment Results ---")
    print(f"1. Training time: {training_duration_seconds:.2f} seconds (~{training_duration_minutes:.2f} minutes).")
    print(f"\n2. The longest token in the vocabulary is: '{longest_token}'")
    print(f"   Length: {len(longest_token)} characters.")
    print("\n3. The full tokenizer model has been saved to '{}'.".format(model_file))
    print("   Vocabulary and merges have been saved to '{}' and '{}' for comparison.".format(vocab_file, merges_file))

if __name__ == "__main__":
    main()
