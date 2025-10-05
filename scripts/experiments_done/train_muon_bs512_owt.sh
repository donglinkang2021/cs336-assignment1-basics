uv run train_muon.py --multirun \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=512 \
    training.max_iters=2500 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4 \
    optimizer.warmup_iters=100,500,1000 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py --multirun \
    data.path=data/openwebtext \
    data.tokenizer_path=hf_tokenizer/openwebtext/tokenizer.json \
    training.batch_size=512 \
    training.max_iters=2500 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4 \
    optimizer.warmup_iters=250 \
    'logger.run_name=owt-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'
