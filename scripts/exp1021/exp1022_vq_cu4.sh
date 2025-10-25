# 增大离散性，使用过更多的codebook
CUDA_VISIBLE_DEVICES=4 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=4,8,16,32 \
    +model.codebook_size=32 \
    training.batch_size=128 \
    model.ffn_type=mhsilu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1021-ts-mhsilu-nc${model.num_codebook}-cs${model.codebook_size}'