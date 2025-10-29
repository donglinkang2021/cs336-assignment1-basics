WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,64]","[64,32]","[128,16]","[16,128]" \
    training.batch_size=128 \
    model.ffn_type=dyna_swiglu \
    model.d_model=768 \
    model.d_ff=2048 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-dyna_swiglu-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[64,64]" \
    training.batch_size=128 \
    model.ffn_type=dyna_swiglu \
    model.d_model=768 \
    model.d_ff=4096 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-dyna_swiglu-cd${model.code_dims}'
