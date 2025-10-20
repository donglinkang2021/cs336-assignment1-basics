# CS336 Spring 2025 Assignment 1: Basics

> ✅ **Status**: Assignment completed with all implementations and experiments finished.
> 
> 📊 **Validation Loss**: 1.33 on TinyStories | 3.33508 on OpenWebText (Leaderboard)

This repository contains my complete implementation for [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics/), including a transformer-based language model built from scratch, BPE tokenizer, training infrastructure, and extensive experiments on model training and optimization.

## 📚 Documentation

- **Assignment PDF**: [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)
- **My Detailed Writeup**: [writup.pdf](./writup.pdf) (Try to answer all questions in the assignment)
- **Experiment Changelog**: [docs/CHANGELOG.md](./CHANGELOG.md)
- **Others**: [docs/](./docs/) (Quick reference extracted from assignment PDF)

## 📋 Overview

This assignment implements a complete language modeling pipeline with the following components:

1. **BPE Tokenizer** ([`cs336_basics/bpe.py`](cs336_basics/bpe.py))
   - Byte-Pair Encoding training with configurable vocabulary size
   - Efficient encoding/decoding with regex-based pre-tokenization

2. **Transformer Architecture** ([`cs336_basics/model.py`](cs336_basics/model.py))
   - Custom implementations: `Linear`, `Embedding`, `RMSNorm`
   - Rotary Position Embeddings (RoPE)
   - SwiGLU activation function
   - Multi-head causal self-attention with KV-cache support
   - Pre-norm decoder-only transformer blocks

3. **Training Infrastructure**
   - Custom optimizers: SGD, AdamW ([`cs336_basics/optimizer.py`](cs336_basics/optimizer.py))
   - Muon optimizer integration ([`train_muon.py`](train_muon.py))
   - Cosine decay learning rate scheduling with warmup
   - Gradient clipping and cross-entropy loss ([`cs336_basics/nn_utils.py`](cs336_basics/nn_utils.py))
   - Multi-backend logging: WandB, TensorBoard, SwanLab ([`cs336_basics/logger.py`](cs336_basics/logger.py))

4. **Text Generation** ([`cs336_basics/generate.py`](cs336_basics/generate.py))
   - Autoregressive generation with top-p (nucleus) sampling
   - KV-cache optimization for efficient inference
   - Generation testing and benchmarking ([`check_gen.py`](check_gen.py))

5. **Comprehensive Experiments** ([`scripts/`](scripts/))
   - Learning rate tuning on TinyStories and OpenWebText
   - Batch size optimization studies
   - Ablation studies: RoPE, RMSNorm, SwiGLU, Pre-norm
   - Cross-dataset training comparisons (main experiment)
   - Leaderboard submission experiments

## 📁 Project Structure

```
cs336_basics/
├── bpe.py              # BPE tokenizer implementation
├── model.py            # Transformer components (Linear, Embedding, RMSNorm, SwiGLU, RoPE, Attention, TransformerLM)
├── nn_utils.py         # Loss functions and utilities (cross_entropy, gradient_clipping)
├── optimizer.py        # SGD, AdamW, Muon optimizers
├── data.py             # Data loading utilities
├── generate.py         # Text generation with KV-cache
├── checkpoint.py       # Model checkpointing
├── logger.py           # Multi-backend logging (WandB, TensorBoard, SwanLab)
└── config.py           # Configuration dataclasses

train.py                # Main training script with Hydra config
train_muon.py          # Training with Muon optimizer
check_gen.py           # Generation testing and benchmarking

conf/                   # Hydra configuration files
scripts/                # Experiment scripts
data_utils/             # Data downloading and tokenization scripts
docs/                   # Implementation notes (extracted from assignment PDF)
```

## 🚀 Setup

### Environment

We use `uv` for environment management. Install it via:

```sh
pip install uv
# or
brew install uv
```

Run any Python file with automatic environment setup:
```sh
uv run <python_file_path>
```

### Testing

Run all unit tests:
```sh
uv run pytest
```

Run specific test categories:
```sh
uv run pytest -k test_linear
uv run pytest -k test_bpe
uv run pytest -k test_transformer
```

### Data Setup

Download TinyStories and OpenWebText datasets:

```sh
bash data_utils/download_dataset.sh
```

Data structure:
```bash
data/
├── TinyStoriesV2-GPT4-train.txt  (2.1G)
├── TinyStoriesV2-GPT4-valid.txt  (21M)
├── owt_train.txt                  (11G)
└── owt_valid.txt                  (277M)
```

Tokenize datasets (see [`data_utils/`](data_utils/) for scripts):
```bash
data/
├── tinystories/    # Tokenized OpenWebText (vocab_size=10000)
├── openwebtext/   # Tokenized OpenWebText (vocab_size=1000)
├── openwebtext-32k/    # Tokenized OpenWebText (vocab_size=32000)
└── ...
```

> [!NOTE] Tokenizer Implementation
>
> While this assignment implements a custom BPE tokenizer from scratch ([`cs336_basics/bpe.py`](cs336_basics/bpe.py)) that passes all unit tests (`uv run pytest`), the actual dataset tokenization for training experiments uses HuggingFace's tokenizer library for efficiency and reliability on large datasets (TinyStories and OpenWebText). Tokenizers are saved in `hf_tokenizer/` directory.

## 🏃 Training

### Basic Training

Train on TinyStories with default config:
```sh
uv run train.py
```

Train on OpenWebText:
```sh
uv run train.py \
    model.vocab_size=32000 \
    data.path=data/openwebtext-32k \
    data.tokenizer_path=hf_tokenizer/openwebtext-32k/tokenizer.json \
    training.batch_size=128 \
    optimizer.max_lr=1e-2
```

### Training with Muon Optimizer

```sh
uv run train_muon.py \
    model.vocab_size=32000 \
    data.path=data/openwebtext-32k
```

### Configuration

Training is configured via Hydra configs in [`conf/`](conf/):
- `train_config.yaml` - Main training configuration
- `model/` - Model architecture configs
- `data/` - Dataset configs
- `optimizer/` - Optimizer configs
- `logger/` - Logging backend configs

Override any config via command line:
```sh
uv run train.py model.d_model=512 optimizer.max_lr=3e-4 training.batch_size=64
```

## 🧪 Experiments

All experiments are tracked using **Weights & Biases** with comprehensive logging of:
- Training/validation losses
- Learning rates and gradient norms
- Entropy and perplexity metrics
- Wallclock time/Relative time (Process)

### Experiment Overview

| Experiment | W&B Report | Description |
|------------|-----------|-------------|
| **Learning Rate** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/jhz7fp86) | Tune learning rate on TinyStories and OpenWebText datasets |
| **Batch Size** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/ejo2bn9n) | Impact of batch size on training performance (TinyStories) |
| **Ablation Studies** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/c54zgdnw) | Component analysis: SwiGLU, RoPE, RMSNorm, Pre-norm (TinyStories) |
| **Main** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/k1na9uic) | Loss comparison between TinyStories and OpenWebText training |
| **Muon** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/p4xh8d7y) | Using Muon for better training performance (OpenWebText) |
| **Leaderboard** | [Link](https://api.wandb.ai/links/donglinkang2021-beijing-institute-of-technology/yjlpxerm) | Final model training and leaderboard submission |

### Key Findings

- ✅ **Validation Loss**: Achieved **1.33** on TinyStories (<1.45, meets requirement)
- 🎯 **Optimal Learning Rate**: **0.01** through comprehensive hyperparameter search
- 🚀 **Optimal Batch Size**: **128** achieves best validation performance in 8.75 minutes (10,000 iterations)
- 📊 **Component Impact**: Ablation studies show importance of RoPE, RMSNorm, SwiGLU, and Pre-norm
- 🏆 **Leaderboard**: Validation loss of **3.33508** on OpenWebText

### Running Experiments

> If you want to reproduce my results here, please ensure you have set up the data and environment as described above. 

Experiment scripts are located in [`scripts/`](scripts/):

```sh
# TinyStories experiments
bash scripts/tinystories_learning_rate.sh       # Learning rate tuning
bash scripts/tinystories_batch_size.sh          # Batch size experiments
bash scripts/tinystories_ablation.sh            # Ablation studies

# OpenWebText experiments
bash scripts/openwebtext_learning_rate.sh       # Learning rate tuning
bash scripts/openwebtext_muon.sh                # Muon optimizer training

# Sync logs to WandB
bash scripts/wandb_sync.sh
```

## 🔍 Text Generation

Test generation quality and performance:
```sh
uv run check_gen.py
```

This script:
- Loads trained checkpoints
- Generates text samples with different prompts
- Measures generation speed (tokens/sec) and memory usage
- Compares performance with/without KV-cache optimization

## 📊 Results

For detailed experimental results, analysis, and comprehensive writeup, see:
- **Main Writeup Repository**: [donglinkang2021/cs336-assignment1-writeup](https://github.com/donglinkang2021/cs336-assignment1-writeup)
  - Complete LaTeX report with all experiments
  - Plotting scripts for visualization ([`code/plot_*.py`](https://github.com/donglinkang2021/cs336-assignment1-writeup/tree/main/code))
  - Experiment results data ([`exps/`](https://github.com/donglinkang2021/cs336-assignment1-writeup/tree/main/exps))

### Document Structure

The writeup is organized as follows:
- **Section 2**: BPE Tokenizer Implementation
- **Section 3**: Transformer Architecture
- **Section 4**: Language Model Training Objectives
- **Section 5**: Training Loop Implementation
- **Section 6**: Text Generation Methods
- **Section 7**: Comprehensive Experimental Results
- **Appendix**: Additional implementation details and code snippets

## 📦 Submission

Create submission package:
```sh
bash make_submission.sh
```

This generates `cs336-spring2025-assignment-1-submission.zip` with all code and test results.

## 📖 More

- Assignment Repository: [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics/)
- Assignment Handout: [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)
- Writeup Repository: [donglinkang2021/cs336-assignment1-writeup](https://github.com/donglinkang2021/cs336-assignment1-writeup)
- Experiment Changelog: [CHANGELOG.md](./CHANGELOG.md)

## 🙏 Acknowledgments

Thanks to the CS336 teaching staff for this comprehensive assignment and leaderboard! Special thanks for providing the infrastructure and test suite that made this learning experience possible.