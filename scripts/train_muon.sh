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