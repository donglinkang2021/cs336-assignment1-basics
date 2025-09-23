# Changelog

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
