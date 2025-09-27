import torch
from typing import Iterable

# uv run pytest -k test_softmax_matches_pytorch
def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    # [..., d_model] -> [..., d_model]
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x_exp / x_exp.sum(dim=dim,keepdim=True)

# uv run pytest -k test_cross_entropy
def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    # (batch_size, num_classes), (batch_size,) -> scalar
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values  # for numerical stability
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()

def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# uv run pytest -k test_gradient_clipping  
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = grad_norm(parameters)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

def compute_entropy_chunked(logits:torch.Tensor, chunk_size:int=128) -> torch.Tensor:
    """Memory-efficient implementation of `compute_entropy`."""
    num_chunks = (logits.shape[1] + chunk_size - 1) // chunk_size
    entropy_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.shape[1])
        chunk_logits = logits[:, start_idx:end_idx, :]
        # Use the numerically stable method for torch.bfloat16, do not use logsumexp
        chunk_probs = chunk_logits.softmax(dim=-1)
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_entropy = -(chunk_probs * chunk_log_probs).sum(dim=-1)
        entropy_chunks.append(chunk_entropy)
    return torch.cat(entropy_chunks, dim=1)
