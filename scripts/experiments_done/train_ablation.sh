uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    'logger.run_name=baseline'

uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    model.ffn_type=silu \
    model.d_ff=2048 \
    'logger.run_name=baseline-swiglu2silu'

uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    model.use_post_norm=true \
    'logger.run_name=baseline-prenorm2postnorm'

uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    model.remove_rmsnorm=true \
    'logger.run_name=baseline-wo_rmsnorm'

uv run train.py \
    training.batch_size=32 \
    training.max_iters=40000 \
    model.remove_rope=true \
    'logger.run_name=baseline-wo_rope'