from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch import Tensor

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

    
# uv run pytest -k test_adamw
class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.01
    ):        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Decoupled weight decay
                p.data.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # Update parameters
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    This implementation is a simplified, non-distributed version.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if p.dim() < 2:
                    # Warning from original paper: do not use for 0D/1D params
                    # Fallback to simple SGD update without momentum for these
                    p.add_(grad, alpha=-lr)
                    continue

                # Shape-adaptive learning rate and weight decay
                eff_lr = lr * max(1, p.size(-2) / p.size(-1)) ** 0.5
                eff_weight_decay = lr * weight_decay

                # Apply weight decay
                p.mul_(1 - eff_weight_decay)

                # Momentum
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()
                
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad) # Nesterov-style momentum update
                
                # Orthogonalize the update direction
                # Use bfloat16 as suggested for stability
                ortho_update = zeropower_via_newtonschulz5(buf.to(torch.bfloat16), ns_steps).to(p.dtype)
                
                # Apply the update
                p.add_(ortho_update, alpha=-eff_lr)

        return loss

# uv run pytest -k test_get_lr_cosine_schedule
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        # Warm-up phase
        return (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        # Cosine annealing phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cos_term = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cos_term * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing phase
        return min_learning_rate