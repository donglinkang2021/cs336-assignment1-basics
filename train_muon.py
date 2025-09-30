import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tokenizers import Tokenizer

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping, compute_entropy_chunked
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.generate import generate
from cs336_basics.logger import Logger
from cs336_basics.config import TrainConfig

@torch.no_grad()
def evaluate(model:TransformerLM, data, cfg: TrainConfig, device):
    """
    Estimates the loss over a number of batches.
    """
    model.eval()
    losses = []
    entropies = []
    for k in tqdm(range(cfg.training.eval_iters), desc="Evaluating", leave=False):
        x, y = get_batch(data, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
        entropies.append(compute_entropy_chunked(logits).mean().item())
    model.train()
    mean_loss = np.mean(losses)
    return {
        'val/loss': mean_loss,
        'val/ppl': np.exp(mean_loss),
        'val/entropy': np.mean(entropies)
    }


@hydra.main(config_path="conf", config_name="train_config", version_base=None)
def main(cfg: TrainConfig) -> None:
    """
    Main training loop managed by Hydra.
    """
    # print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))
    # return
    # --- Setup ---
    logger = Logger(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Output directory: {output_dir}")
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    # --- Model and Optimizer ---
    model = TransformerLM(**cfg.model).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.max_lr)
    
    start_iter = 0
    # --- Checkpoint Loading ---
    if cfg.training.resume_from:
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        start_iter = load_checkpoint(cfg.training.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()
    for it in tqdm(range(start_iter, cfg.training.max_iters), desc="Training"):
        # Learning rate schedule
        lr = get_lr_cosine_schedule(it, cfg.optimizer.max_lr, cfg.optimizer.min_lr, cfg.optimizer.warmup_iters, cfg.training.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get a batch of data
        x, y = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient Clipping
        grad_norm = gradient_clipping(model.parameters(), max_l2_norm=1.0)

        optimizer.step()

        # --- Logging ---
        if it % cfg.training.log_interval == 0 or it == cfg.training.max_iters - 1:
            duration = time.time() - start_time
            ent = compute_entropy_chunked(logits).mean()
            tqdm.write(f"Iter {it}: Train loss={loss.item():.4f}, LR={lr:.6f}, Time={duration:.2f}s")
            logger.log_metrics({
                'train/loss': loss.item(), 
                'train/ppl': loss.exp().item(),
                'train/lr': lr,
                'train/entropy': ent.item(),
                'train/grad_norm': grad_norm
            }, step=it)
            
        # --- Evaluation and Checkpointing ---
        if it > 0 and (it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1):
            metrics = evaluate(model, val_data, cfg, device)
            tqdm.write(f"Iter {it}: Val loss={metrics['val/loss']:.4f}")
            logger.log_metrics(metrics, step=it)
            
            checkpoint_path = output_dir / f'ckpt_{it}.pt'
            tqdm.write(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, it, checkpoint_path)
    
    tqdm.write("Training finished.")
    
    # --- Generation ---
    tokenizer_path = cfg.data.tokenizer_path
    tokenizer = Tokenizer.from_file(tokenizer_path)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Beginning generation...")
    generated_output = tokenizer.decode(
        generate(
            model, 
            context, 
            max_new_tokens=1000, 
            block_size=cfg.model.context_length,
            temperature=0.6,
            top_p=0.95
        )[0].tolist()
    )    
    tqdm.write("\n--- Generated Text ---")
    tqdm.write(generated_output)
    # Log generated text
    logger.log_text("Generated Text", generated_output, step=cfg.training.max_iters)
    logger.close()
    OmegaConf.save(cfg, output_dir / 'config.yaml')


if __name__ == "__main__":
    main()
