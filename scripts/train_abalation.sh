uv run train.py \
    training.batch_size=64 \
    training.max_iters=20000 \
    optimizer.max_lr=3e-4 \
    optimizer.warmup_iters=500 \
    'logger.run_name=abalation_baseline'
