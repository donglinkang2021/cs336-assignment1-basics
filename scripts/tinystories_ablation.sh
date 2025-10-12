uv run train.py \
    training.batch_size=128 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts'

uv run train.py \
    training.batch_size=128 \
    model.ffn_type=silu \
    model.d_ff=2048 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts-swiglu2silu'

uv run train.py \
    training.batch_size=128 \
    model.use_post_norm=true \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts-prenorm2postnorm'

uv run train.py \
    training.batch_size=128 \
    model.remove_rmsnorm=true \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts-wo_rmsnorm'

uv run train.py \
    training.batch_size=128 \
    model.remove_rope=true \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts-wo_rope'
