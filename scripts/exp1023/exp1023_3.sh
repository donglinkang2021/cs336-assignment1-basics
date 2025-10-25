WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.head_dim=768 \
    training.batch_size=128 \
    model.ffn_type=head \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1023-ts-head-dm${model.d_model}-dff${model.d_ff}-hs${model.head_dim}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.head_dim=8,16,32,64,128,512,1024 \
    training.batch_size=128 \
    model.ffn_type=head \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1023-ts-head-dm${model.d_model}-dff${model.d_ff}-hs${model.head_dim}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.head_dim=128 \
    training.batch_size=128 \
    model.ffn_type=head \
    model.d_model=768 \
    model.d_ff=256,512,768,1024,2048 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1023-ts-head-dm${model.d_model}-dff${model.d_ff}-hs${model.head_dim}'
