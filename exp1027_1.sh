WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=2 \
    +model.code_dim=4 \
    training.batch_size=128 \
    model.ffn_type=cache_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-ts-cache_silu-nc${model.num_codebook}-dc${model.code_dim}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=2 \
    +model.code_dim=8,16,32,64,128,256 \
    training.batch_size=128 \
    model.ffn_type=cache_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-ts-cache_silu-nc${model.num_codebook}-dc${model.code_dim}'
