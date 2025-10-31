WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,96]","[64,48]" \
    +model.outer_norm=True,False \
    +model.num_codebook=2,4,8 \
    training.batch_size=128 \
    model.ffn_type=dynam_silu,outerm_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1030-${model.ffn_type}-norm${model.outer_norm}-nc${model.num_codebook}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[64,64]" \
    +model.outer_norm=True,False \
    +model.num_codebook=2,4,8 \
    training.batch_size=128 \
    model.ffn_type=dynam_silu,outerm_silu \
    model.d_model=768 \
    model.d_ff=4096 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1030-${model.ffn_type}-norm${model.outer_norm}-nc${model.num_codebook}-cd${model.code_dims}'