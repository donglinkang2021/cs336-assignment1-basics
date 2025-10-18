CUDA_VISIBLE_DEVICES=5 uv run train_qknorm.py -m \
    training.batch_size=128 \
    optimizer.max_lr=1e-2,1e-3,3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts-lr${optimizer.max_lr}-qknorm'
