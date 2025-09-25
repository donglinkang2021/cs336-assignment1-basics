import torch

# uv run pytest -k test_softmax_matches_pytorch
def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    # [..., d_model] -> [..., d_model]
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x_exp / x_exp.sum(dim=dim,keepdim=True)
