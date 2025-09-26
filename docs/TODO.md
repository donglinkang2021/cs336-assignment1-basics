# todo

1. Train a BPE tokenizer on the TinyStories dataset.
2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.
3. Train a Transformer LM on the TinyStories dataset.
4. Generate samples and evaluate perplexity using the trained Transformer LM.
5. Train models on OpenWebText and submit your attained perplexities to a leaderboard.

## Milestone: Tokenizer

```bash
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
```

## Milestone: Transformer LM

```bash
uv run pytest tests/test_model.py
```

```bash
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_silu
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
```

## Milestone: Train

```bash
uv run pytest -k test_cross_entropy
uv run pytest -k test_adamw
uv run pytest -k test_get_lr_cosine_schedule
uv run pytest -k test_gradient_clipping
uv run pytest -k test_get_batch
```