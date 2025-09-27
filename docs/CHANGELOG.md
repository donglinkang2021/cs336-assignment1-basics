# Changelog

## [0.0.5] - 20250927

- [code] add training script, integrating with logger, add the default config, need tokenizer to tokenize the txt before

## [0.0.4] - 20250926

- [code] pass all the tests, pass `uv run pytest`, fix all the warnings
- [code] checkpointing, pass `test_checkpointing`
- [code] data loader, pass `test_get_batch`, `uv run pytest tests/test_data.py`

### Record

终于全部过完了所有的test cases，下一步的计划是打算开始做用wandb训练和generate的部分了；

```bash
(cs336-basics) [root:assignment1-basics]$ uv run pytest
==================================== test session starts ====================================
tests/test_data.py::test_get_batch PASSED
tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
tests/test_model.py::test_swiglu PASSED
tests/test_model.py::test_scaled_dot_product_attention PASSED
tests/test_model.py::test_4d_scaled_dot_product_attention PASSED
tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED
tests/test_model.py::test_transformer_block PASSED
tests/test_model.py::test_rmsnorm PASSED
tests/test_model.py::test_rope PASSED
tests/test_model.py::test_silu_matches_pytorch PASSED
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED
tests/test_nn_utils.py::test_cross_entropy PASSED
tests/test_nn_utils.py::test_gradient_clipping PASSED
tests/test_optimizer.py::test_adamw PASSED
tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED
tests/test_serialization.py::test_checkpointing PASSED
tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is expected to take more memory than allotted (1MB).)
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
============================== 47 passed, 1 xfailed in 43.99s ===============================
```

## [0.0.3] - 20250925

- [handout] 35/50
- [code] `uv run pytest tests/test_optimizer.py`
- [code] `uv run pytest tests/test_nn_utils.py`
- [code] gradient clipping, pass `test_gradient_clipping`
- [code] learning rate scheduler, pass `test_get_lr_cosine_schedule`
- [code] adamw, pass `test_adamw`
- [code] cross entropy, pass `test_cross_entropy`
- [code] pass all the test cases in `uv run pytest tests/test_model.py`
- [code] transformer LM, pass `test_transformer_lm`
- [code] transformer block, pass `test_transformer_block`
- [code] softmax, scaled_dot_product_attention, multihead_self_attention, pass the test cases
- [code] rope, pass `test_rope`
- [code] swiglu FFN layer, pass `test_swiglu`
- [code] rmsnorm layer, pass `test_rmsnorm`
- [code] linear/embedding layer, pass `test_linear/test_embedding`

### Record

终于全部都过了，自己在Attention部分参考`nanovllm`的`qwen3`写法来写集成的rope -> [here](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/models/qwen3.py#L53)，目前结果如下：

```bash
(cs336-basics) [root:assignment1-basics]$ uv run pytest tests/test_model.py
============================== test session starts ==============================
tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
tests/test_model.py::test_swiglu PASSED
tests/test_model.py::test_scaled_dot_product_attention PASSED
tests/test_model.py::test_4d_scaled_dot_product_attention PASSED
tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED
tests/test_model.py::test_transformer_block PASSED
tests/test_model.py::test_rmsnorm PASSED
tests/test_model.py::test_rope PASSED
tests/test_model.py::test_silu_matches_pytorch PASSED
============================== 13 passed in 2.28s ===============================
(cs336-basics) [root:assignment1-basics]$ uv run pytest tests/test_nn_utils.py
============================== test session starts ==============================
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED
tests/test_nn_utils.py::test_cross_entropy PASSED
tests/test_nn_utils.py::test_gradient_clipping PASSED
=============================== 3 passed in 2.01s ===============================
(cs336-basics) [root:assignment1-basics]$ uv run pytest tests/test_optimizer.py
============================== test session starts ==============================
tests/test_optimizer.py::test_adamw PASSED
tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED
=============================== 2 passed in 3.04s ===============================
```

## [0.0.2] - 20250924

- [code] tokenizer, pass the `tests/test_tokenizer.py`
- [code] pre-tokenization + multiprocessing, pass the `tests/train_bpe.py`
- [code] pre-tokenization, use `regex.split` and `regex.finditer` to handle special tokens and pre-tokenization, make it to ` 2.282496929168701 < 1.5`, but still too slow

### Record

看了一下 `gpt2_bytes_to_unicode` 的实现，他的主要原理就是把 0-255中的 unprintable 的字符全部去掉，将可以打印的 0-255 的字符全部放在前面从0开始编码（一共188个可打印的），后面的68个字符就把 256+i 的编码塞到前面去；

关于tokenizer的基本全过了，第一个train_bpe的部分借鉴了csdn上一位老哥的代码[here](refer: https://blog.csdn.net/Bug_makerACE/article/details/149248369)，第二部分更多是参考了karpathy的minbpe/regex那的实现来优化encode的速度 👉 [karpathy/minbpe](https://github.com/karpathy/minbpe)；

目前结果如下：

```bash
(cs336-basics) [root:assignment1-basics]$ uv run pytest tests/test_train_bpe.py 
============================== test session starts ==============================
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
============================== 3 passed in 21.30s ===============================
(cs336-basics) [root:assignment1-basics]$ uv run pytest tests/test_tokenizer.py
============================== test session starts ==============================
tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
...
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is expected to take more memory than allotted (1MB).)
======================== 24 passed, 1 xfailed in 14.16s =========================
```

## [0.0.1] - 20250923

- [code] implement the simple version of bpe refer to the minbpe of karpathy, but fail 3/3 cases
- [handout] bpe tokenizer training

### Record

开始正式看了一下这个repo, 运行下面的代码看看效果

```bash
uv run pytest
```

```bash
=============================== short test summary info ================================
FAILED tests/test_data.py::test_get_batch - NotImplementedError
FAILED tests/test_model.py::test_linear - NotImplementedError
...
======================= 47 failed, 1 xfailed in 86.94s (0:01:26) =======================
```

测试了一下发现太慢了，问了gemini可以先测试某部分 --> [部分测试加快速度](../tips/部分测试加快速度.md)
