# uv run train.py \
#     +model.add_qknorm=True \
#     +model.tie_embeddings=True \
#     training.batch_size=128 \
#     model.ffn_type=silu \
#     model.d_model=768 \
#     model.d_ff=3072 \
#     optimizer.max_lr=1e-2 \
#     optimizer.weight_decay=0.01 \
#     optimizer.max_l2_norm=2.0 \
#     optimizer.betas="[0.95,0.999]" \
#     'logger.run_name=exp1021-ts-silu'

# uv run train.py \
#     +model.add_qknorm=True \
#     +model.tie_embeddings=True \
#     training.batch_size=128 \
#     model.ffn_type=swiglu \
#     model.d_model=768 \
#     model.d_ff=2048 \
#     optimizer.max_lr=1e-2 \
#     optimizer.weight_decay=0.01 \
#     optimizer.max_l2_norm=2.0 \
#     optimizer.betas="[0.95,0.999]" \
#     'logger.run_name=exp1021-ts-swiglu'

uv run train.py \
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

uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.num_codebook=4,8,16 \
    +model.codebook_size=32,64,128,256,512,1024,2048,3072 \
    training.batch_size=128 \
    model.ffn_type=mhvq \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1021-ts-vq-nc${model.num_codebook}-cs${model.codebook_size}'
