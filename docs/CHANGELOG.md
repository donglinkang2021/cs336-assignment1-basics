# Changelog

## [0.0.2] - 20250924

- [code] pre-tokenization, use `regex.split` and `regex.finditer` to handle special tokens and pre-tokenization, make it to ` 2.282496929168701 < 1.5`, but still too slow

### Record

看了一下 `gpt2_bytes_to_unicode` 的实现，他的主要原理就是把 0-255中的 unprintable 的字符全部去掉，将可以打印的 0-255 的字符全部放在前面从0开始编码（一共188个可打印的），后面的68个字符就把 256+i 的编码塞到前面去；

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
