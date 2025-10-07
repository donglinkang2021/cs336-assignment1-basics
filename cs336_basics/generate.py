import torch
from .nn_utils import softmax
from .model import KVCache, TransformerLM

def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True) # from high to low
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find a smallest set, let its sum >= top_p
    # cumsum_probs - sorted_probs means the all-left-probs sum
    mask = cumsum_probs - sorted_probs > top_p 
    sorted_probs[mask] = 0.0
    # Re-normalize the probabilities
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    # Map back to original indices
    next_token = torch.gather(sorted_indices, 1, next_token)
    return next_token

@torch.inference_mode()
def generate(
    model:TransformerLM, 
    idx:torch.Tensor,
    max_new_tokens:int, 
    block_size:int = None,
    temperature:float = 1.0,
    top_p:float = 1.0,
    use_kv_cache: bool = False
) -> torch.Tensor:
    # idx is (B, T) array of indices in the current context
    for i in range(max_new_tokens):        
        if use_kv_cache:
            if i == 0:
                token_positions = torch.arange(idx.size(1), device=idx.device)
                logits = model(idx, token_positions)
            else:
                token_positions = torch.tensor([idx.size(1) - 1], device=idx.device)
                logits = model(idx[:, -1:], token_positions)  # only process the last token
        else:
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] if block_size else idx        
            logits = model(idx_cond) # (B, T, C)
        
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        
        if temperature == 0.0:
            # greedy decoding: always pick the most likely token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
        else:
            # apply temperature
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = softmax(logits, dim=-1) # (B, C)
            # apply top-p (nucleus) sampling
            idx_next = sample_top_p(probs, top_p) # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

from .model import get_rope

def install_kv_cache(model: TransformerLM, batch_size: int, total_len: int):
    """Install KV cache in all attention layers of the model"""
    for layer in model.layers:
        layer_dtype = layer.attn.q_proj.weight.dtype
        layer_device = layer.attn.q_proj.weight.device
        layer.attn.cache = KVCache(
            batch_size=batch_size,
            seq_length=total_len,
            n_kv_heads=layer.attn.num_heads,
            head_dim=layer.attn.head_dim,
            dtype=layer_dtype,
            device=layer_device,
        )
        # Attention!! need to update rope max_len
        # otherwise it will cause
        if layer.attn.rope.max_seq_len < total_len:
            layer.attn.rope = get_rope(
                theta = layer.attn.theta,
                d_k = layer.attn.head_dim,
                max_seq_len = total_len,
            ).to(device=layer_device, dtype=layer_dtype)
        
def remove_kv_cache(model: TransformerLM):
    """Remove KV cache in all attention layers of the model"""
    for layer in model.layers:
        layer_dtype = layer.attn.q_proj.weight.dtype
        layer_device = layer.attn.q_proj.weight.device
        layer.attn.cache = None
        layer.attn.rope = get_rope(
            theta = layer.attn.theta,
            d_k = layer.attn.head_dim,
            max_seq_len = model.max_seq_len,
        ).to(device=layer_device, dtype=layer_dtype)
