WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    training.batch_size=128 \
    model.ffn_type=vq \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1021-ts-vq'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=2 \
    +model.codebook_size=64 \
    training.batch_size=128 \
    model.ffn_type=mhvq \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1021-ts-mhvq-nc${model.num_codebook}-cs${model.codebook_size}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=4 \
    +model.codebook_size=32 \
    training.batch_size=128 \
    model.ffn_type=mhvq \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1021-ts-mhvq-nc${model.num_codebook}-cs${model.codebook_size}'
