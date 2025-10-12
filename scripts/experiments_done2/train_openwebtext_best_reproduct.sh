export WANDB_MODE=offline
uv run train_muon.py \
    model.vocab_size=32000 \
    model.num_layers=8 \
    model.d_model=768 \
    model.d_ff=2048 \
    data.path=data/openwebtext-32k \
    data.tokenizer_path=hf_tokenizer/openwebtext-32k/tokenizer.json \
    training.batch_size=64 \
    training.max_iters=20000 \
    optimizer.warmup_iters=1000 \
    optimizer.max_lr=3e-4 \
    optimizer.min_lr=0 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=0.5 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=owtlb3-muon-bs${training.batch_size}-lr${optimizer.max_lr}-wd${optimizer.weight_decay}-norm${optimizer.max_l2_norm}-betas${optimizer.betas}'

# GPT-2 small equivalent model
uv run train.py \
    model.vocab_size=32000 \
    model.num_layers=12 \
    model.num_heads=12 \
    model.d_model=768 \
    model.d_ff=2048 \
    data.path=data/openwebtext-32k \
    data.tokenizer_path=hf_tokenizer/openwebtext-32k/tokenizer.json \
    training.batch_size=128 \
    training.max_iters=20000 \
    optimizer.warmup_iters=1000 \
    optimizer.max_lr=3e-4 \
    optimizer.min_lr=0 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=0.5 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=owtlb4-bs${training.batch_size}-lr${optimizer.max_lr}-wd${optimizer.weight_decay}-norm${optimizer.max_l2_norm}-betas${optimizer.betas}'

uv run train.py -m \
    model.vocab_size=32000 \
    model.num_layers=12 \
    model.num_heads=12 \
    model.d_model=768 \
    model.d_ff=2048 \
    data.path=data/openwebtext-32k \
    data.tokenizer_path=hf_tokenizer/openwebtext-32k/tokenizer.json \
    training.batch_size=64 \
    training.max_iters=20000 \
    optimizer.warmup_iters=1000 \
    optimizer.max_lr=1e-3,5e-4 \
    optimizer.min_lr=0 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=0.5 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=owtlb5-bs${training.batch_size}-lr${optimizer.max_lr}-wd${optimizer.weight_decay}-norm${optimizer.max_l2_norm}-betas${optimizer.betas}'
