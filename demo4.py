import torch
from cs336_basics.model import TransformerLM

base_config = dict(
    vocab_size = 10000,
    context_length = 256,
    d_model = 768,
    num_layers = 4,
    num_heads = 16,
    d_ff = 3072,
    rope_theta = 10000.0,
)

base_w_silu = TransformerLM(
    **base_config,
    ffn_type="silu",
    tie_embeddings = True,
)

base_w_outersilu = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="outer_silu",
    code_dims=[48, 64],
)

base_w_dynasilu = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="dyna_silu",
    code_dims=[48, 64],
)

B, T = 128, 256
x = torch.randint(0, 10000, (B, T))
print(x[:5,:10])

out1 = base_w_silu(x)
out2 = base_w_outersilu(x)
out3 = base_w_dynasilu(x)

print(f"silu        - mean: {out1.mean():.4f}, std: {out1.std():.4f}")
print(f"outer_silu  - mean: {out2.mean():.4f}, std: {out2.std():.4f}")
print(f"dyna_silu   - mean: {out2.mean():.4f}, std: {out2.std():.4f}")