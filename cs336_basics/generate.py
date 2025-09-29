import torch
from .nn_utils import softmax

def generate(
    model:torch.nn.Module, 
    idx:torch.Tensor,
    max_new_tokens:int, 
    block_size:int = None,
    temperature:float = 1.0,
    top_p:float = None
) -> torch.Tensor:
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:] if block_size else idx
        # get the predictions
        logits = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        
        if temperature == 0.0:
            # greedy decoding: always pick the most likely token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # apply temperature
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = softmax(logits, dim=-1) # (B, C)
            # apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                # renormalize
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
