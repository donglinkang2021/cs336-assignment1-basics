import torch
from pathlib import Path
from tokenizers import Tokenizer
from omegaconf import OmegaConf

from cs336_basics.model import TransformerLM
from cs336_basics.generate import generate
from configs.config import TrainConfig

# Load model from checkpoint
# the best model of openwebtext
# ckpt_path = "outputs/multiruns/2025-10-01_18-19-21/2/ckpt_4999.pt"
# the best model of tinystories
ckpt_path = "outputs/runs/2025-10-02_05-14-38/ckpt_19999.pt"
# get config file
cfg_path = Path(ckpt_path).parent / ".hydra" / "config.yaml"
cfg:TrainConfig = OmegaConf.load(cfg_path)
print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerLM(**cfg.model).to(device)
if cfg.training.is_compile:
    model = torch.compile(model)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from {ckpt_path}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params/1e6:.2f}M")

for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.data.dtype, p.device)


model.eval()
tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Beginning generation...")
generated_output = tokenizer.decode(
    generate(
        model, 
        context, 
        max_new_tokens=1000, 
        block_size=cfg.model.context_length,
        temperature=0.6,
    )[0].tolist()
)
print("\n--- Generated Text ---")
print(generated_output)