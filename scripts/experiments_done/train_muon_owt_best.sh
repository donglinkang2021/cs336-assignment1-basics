uv run train_muon.py \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=64 \
    training.max_iters=20000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=2000 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=32 \
    training.max_iters=40000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=4000 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=128 \
    training.max_iters=10000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=1000 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=256 \
    training.max_iters=5000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=500 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'
