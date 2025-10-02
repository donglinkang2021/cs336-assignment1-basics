uv run train_muon.py --multirun \
    training.batch_size=512 \
    training.max_iters=2500 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4,3e-5 \
    optimizer.mm_warmup_steps=100,300,500 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py --multirun \
    training.batch_size=512 \
    training.max_iters=2500 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4,3e-5 \
    optimizer.mm_warmup=false \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-wo_mm_warmup'
