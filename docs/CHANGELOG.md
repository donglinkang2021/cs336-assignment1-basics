# Changelog

## [0.0.3] - 20250925

- [code] swiglu FFN layer, pass `test_swiglu`
- [code] rmsnorm layer, pass `test_rmsnorm`
- [code] linear/embedding layer, pass `test_linear/test_embedding`

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
