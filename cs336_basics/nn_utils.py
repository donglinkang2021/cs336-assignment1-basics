import torch

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
