WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    training.batch_size=256,128,64,32,16,8,4,2,1 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts-bs${training.batch_size}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    training.batch_size=512 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts-bs${training.batch_size}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    training.batch_size=768 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts-bs${training.batch_size}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    training.batch_size=1024 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts-bs${training.batch_size}'