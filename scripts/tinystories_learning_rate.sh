CUDA_VISIBLE_DEVICES=7 uv run train.py -m \
    optimizer.max_lr=1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,6e-3,1e-2,3e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts-lr${optimizer.max_lr}'
