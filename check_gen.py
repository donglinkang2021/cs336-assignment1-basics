import torch
from pathlib import Path
from tokenizers import Tokenizer
from omegaconf import OmegaConf

from cs336_basics.model import TransformerLM
from cs336_basics.generate import generate
from cs336_basics.config import TrainConfig

# Load model from checkpoint
ckpt_path = "/inspire/hdd/global_user/donglinkang-253108120084/standford-cs336/assignment1-basics/outputs/multiruns/2025-09-28_20-56-56/4/ckpt_19999.pt"
# ckpt_path = "/inspire/hdd/global_user/donglinkang-253108120084/standford-cs336/assignment1-basics/outputs/multiruns/2025-09-28_20-36-53/4/ckpt_19999.pt"
# get config file
cfg_path = Path(ckpt_path).parent / ".hydra" / "config.yaml"
cfg:TrainConfig = OmegaConf.load(cfg_path)
print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerLM(**cfg.model).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from {ckpt_path}")

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
        top_p=0.95
    )[0].tolist()
)
print("\n--- Generated Text ---")
print(generated_output)