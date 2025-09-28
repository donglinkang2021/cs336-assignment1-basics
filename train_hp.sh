uv run train.py --multirun \
    training.batch_size=512 \
    training.max_iters=2500 \
    optimizer.max_lr=1e-4,3e-4 \
    optimizer.warmup_iters=100,500,1000 \
    'logger.run_name=gpt-bs${training.batch_size}-ms${training.max_iters}-lr${optimizer.max_lr}-warmup${optimizer.warmup_iters}'

uv run train.py --multirun \
    training.batch_size=256 \
    training.max_iters=5000 \
    optimizer.max_lr=1e-4,3e-4 \
    optimizer.warmup_iters=100,500,1000 \
    'logger.run_name=gpt-bs${training.batch_size}-ms${training.max_iters}-lr${optimizer.max_lr}-warmup${optimizer.warmup_iters}'

uv run train.py --multirun \
    training.batch_size=128 \
    training.max_iters=10000 \
    optimizer.max_lr=1e-4,3e-4 \
    optimizer.warmup_iters=100,500,1000 \
    'logger.run_name=gpt-bs${training.batch_size}-ms${training.max_iters}-lr${optimizer.max_lr}-warmup${optimizer.warmup_iters}'

uv run train.py --multirun \
    training.batch_size=64 \
    training.max_iters=20000 \
    optimizer.max_lr=1e-4,3e-4 \
    optimizer.warmup_iters=100,500,1000 \
    'logger.run_name=gpt-bs${training.batch_size}-ms${training.max_iters}-lr${optimizer.max_lr}-warmup${optimizer.warmup_iters}'
