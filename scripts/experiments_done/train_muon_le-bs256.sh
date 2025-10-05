uv run train_muon.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    'logger.run_name=baseline-w_muon'

uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    training.is_compile=true \
    'logger.run_name=baseline-w_compile'

uv run train_muon.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    training.is_compile=true \
    'logger.run_name=baseline-w_compile-w_muon'

uv run train_muon.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    'logger.run_name=baseline-w_muon'

uv run train_muon.py \
    training.batch_size=64 \
    training.max_iters=20000 \
    training.is_compile=true \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}'

uv run train_muon.py \
    training.batch_size=128 \
    training.max_iters=10000 \
    training.is_compile=true \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}'

uv run train_muon.py \
    training.batch_size=256 \
    training.max_iters=5000 \
    training.is_compile=true \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}'

uv run train_muon.py --multirun \
    training.batch_size=256 \
    training.max_iters=5000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4,3e-5 \
    optimizer.mm_warmup_steps=100,300,500 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py --multirun \
    training.batch_size=256 \
    training.max_iters=5000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2,3e-3,3e-4,3e-5 \
    optimizer.mm_warmup=false \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-wo_mm_warmup'

uv run train_muon.py --multirun \
    training.batch_size=128 \
    training.max_iters=10000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.mm_warmup_steps=100,300,500 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py --multirun \
    training.batch_size=64 \
    training.max_iters=20000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.mm_warmup_steps=100,300,500 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py --multirun \
    training.batch_size=32 \
    training.max_iters=40000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.mm_warmup_steps=100,300,500 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py \
    training.batch_size=64 \
    training.max_iters=20000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=2000 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'

uv run train_muon.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    training.is_compile=true \
    optimizer.muon_lr=3e-2 \
    optimizer.warmup_iters=4000 \
    'logger.run_name=baseline-w_compile-w_muon-bs${training.batch_size}-ms${training.max_iters}-mlr${optimizer.muon_lr}-mmws${optimizer.mm_warmup_steps}'
