# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

``` sh
bash data_utils/download_dataset.sh
```

Our data is stored on the path `data`

```bash
(cs336-basics) [root:assignment1-basics]$ ls -ahl data
total 14G
drwxr-xr-x 2 root root 4.0K Sep  7 14:10 .
drwxr-xr-x 3 root root   24 Sep  8 14:47 ..
-rw-r--r-- 1 root root 2.1G Sep  7 11:11 TinyStoriesV2-GPT4-train.txt
-rw-r--r-- 1 root root  22M Sep  7 11:13 TinyStoriesV2-GPT4-valid.txt
-rw-r--r-- 1 root root  339 Sep  7 11:18 download_dataset.sh
-rw-r--r-- 1 root root  12G Apr  1  2024 owt_train.txt
-rw-r--r-- 1 root root 277M Apr  1  2024 owt_valid.txt
```
