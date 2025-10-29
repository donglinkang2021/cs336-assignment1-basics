WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,96]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=outer_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-outer_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[64,48]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=outer_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-outer_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,64]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=outer_silu \
    model.d_model=768 \
    model.d_ff=2048 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-outer_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[64,64]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=outer_silu \
    model.d_model=768 \
    model.d_ff=4096 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-outer_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,96]","[64,48]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=dyna_silu \
    model.d_model=768 \
    model.d_ff=3072 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-dyna_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[32,64]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=dyna_silu \
    model.d_model=768 \
    model.d_ff=2048 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-dyna_silu-norm${model.outer_norm}-cd${model.code_dims}'

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 uv run train.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    +model.code_dims="[64,64]","[16,16,16]" \
    +model.outer_norm=True,False \
    training.batch_size=128 \
    model.ffn_type=dyna_silu \
    model.d_model=768 \
    model.d_ff=4096 \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=exp1027-dyna_silu-norm${model.outer_norm}-cd${model.code_dims}'
