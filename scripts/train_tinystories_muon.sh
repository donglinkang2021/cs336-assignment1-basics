CUDA_VISIBLE_DEVICES=6 uv run train_muon.py -m \
    optimizer.max_lr=1e-4,3e-4,5e-4,1e-3 \
    optimizer.weight_decay=0.0,0.01,0.1 \
    optimizer.max_l2_norm=0.5,1.0,2.0 \
    optimizer.betas="[0.9,0.95]","[0.9,0.999]","[0.95,0.999]" \
    'logger.run_name=ts-w_muon-bs${training.batch_size}-lr${optimizer.max_lr}-wd${optimizer.weight_decay}-norm${optimizer.max_l2_norm}-betas${optimizer.betas}'
