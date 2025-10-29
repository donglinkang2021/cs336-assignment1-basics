from cs336_basics.model import TransformerLM
from tabulate import tabulate
import collections

def count_parameters(model: TransformerLM):
    return sum(p.numel() for p in model.parameters())

def detailed_count_parameters(model: TransformerLM):
    """Counts parameters for each module, returns a formatted dict."""
    counts = collections.defaultdict(int)
    total_params = count_parameters(model)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'token_embeddings' in name:
            counts['token_embeddings'] += p.numel()
        elif 'attn' in name:
            counts['attn'] += p.numel()
        elif 'ffn' in name:
            counts['ffn'] += p.numel()
        elif 'ln' in name or 'norm' in name: # Catches ln1, ln2, ln_final, RMSNorm
            counts['ln'] += p.numel()
        elif 'lm_head' in name:
            counts['lm_head'] += p.numel()
        else:
            # This helps catch any parameters we might have missed.
            counts['other'] += p.numel()

    # Format the output strings
    formatted_counts = {}
    for name, count in counts.items():
        if total_params > 0:
            percentage = 100 * count / total_params
            formatted_counts[name] = f"{count:,} ({percentage:.1f}%)"
        else:
            formatted_counts[name] = "0 (0.0%)"
    
    formatted_counts['Total'] = f"{total_params:,}"
    return formatted_counts


base_config = dict(
    vocab_size = 10000,
    context_length = 256,
    d_model = 768,
    num_layers = 4,
    num_heads = 16,
    d_ff = 3072,
    rope_theta = 10000.0,
)

base_model = TransformerLM(
    **base_config,
    ffn_type="silu",
)
base_w_tie_embeddings = TransformerLM(
    **base_config,
    ffn_type="silu",
    tie_embeddings = True,
)
base_w_silu1 = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="silu1",
)
base_w_mhsilu1 = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="mhsilu1",
    num_codebook=4,
    codebook_size=32,
) # 3072 = 3 * 1024
base_w_mhsilu = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="mhsilu",
    num_codebook=4,
    codebook_size=32,
)
base_w_outersilu = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="outer_silu",
    code_dims=[96, 32],
)
base_w_dynaswiglu = TransformerLM(
    **base_config,
    tie_embeddings = True,
    ffn_type="dyna_swiglu",
    code_dims=[64, 48],
)

# print(base_w_silu1) # You can uncomment this to see the model structure

# --- New Table Generation Logic ---

all_models = {
    name: obj
    for name, obj in sorted(globals().items())
    if name.startswith("base") and isinstance(obj, TransformerLM)
}

rows = []
headers = ["Model / Module", "Parameters"]
module_order = ['token_embeddings', 'attn', 'ffn', 'ln', 'lm_head', 'Total']

for model_name, model_obj in all_models.items():
    rows.append([f"--- {model_name} ---", "---"])
    
    counts = detailed_count_parameters(model_obj)
    
    for module_name in module_order:
        if module_name in counts:
            # Indent module names for clarity
            display_name = f"  {module_name}"
            rows.append([display_name, counts[module_name]])

print(tabulate(rows, headers=headers, tablefmt="github"))
