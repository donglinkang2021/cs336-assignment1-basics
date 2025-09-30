
import os
from typing import BinaryIO, IO
import torch

# uv run pytest -k test_checkpointing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    if isinstance(optimizer, list):
        optimizer_states = [opt.state_dict() for opt in optimizer]
    else:
        optimizer_states = optimizer.state_dict()    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_states,
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    if isinstance(optimizer, list):
        for opt, state in zip(optimizer, checkpoint["optimizer_state_dict"]):
            opt.load_state_dict(state)
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]