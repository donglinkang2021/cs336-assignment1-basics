import torch
import torch.nn as nn
import math

def init_weights(m:nn.Module):
    if isinstance(m, Linear):
        std = math.sqrt(2.0 / (m.in_features + m.out_features))
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, Embedding):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    elif isinstance(m, RMSNorm):
        nn.init.ones_(m.weight)

# uv run pytest -k test_linear
class Linear(nn.Module):
    """ A simple linear layer implemented from scratch."""
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(
        self, 
        in_features:int, 
        out_features:int, 
        bias:bool=False,
        device=None, 
        dtype=None, 
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., in_features] -> [..., out_features]
        return torch.einsum('...i,oi->...o', x, self.weight) + (
            self.bias if self.bias is not None else 0
        )

# uv run pytest -k test_embedding
class Embedding(nn.Module):
    """ A simple embedding layer implemented from scratch. """
    num_embeddings: int
    embedding_dim: int
    weight: torch.Tensor
    
    def __init__(
        self, 
        num_embeddings:int, 
        embedding_dim:int, 
        device=None, 
        dtype=None
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **kwargs)
        )
    
    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        # [...] -> [..., embedding_dim]
        return self.weight[token_ids]

# uv run pytest -k test_rmsnorm
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Reference https://github.com/donglinkang2021/normalize-layers-pytorch/
    """
    d_model: int
    eps: float
    weight: torch.Tensor
    
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device=None, 
        dtype=None
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, **kwargs))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_out = x * var.rsqrt() * self.weight
        return x_out.to(in_dtype)

# uv run pytest -k test_silu
def silu(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# uv run pytest -k test_swiglu
class SwiGLU(nn.Module):
    """ SwiGLU FFN """
    d_model: int
    d_ff: int
    
    def __init__(self, d_model:int, d_ff:int) -> None:
        super().__init__()
        self.d_model = d_model
        # should be roughly d_ff = 8/3 * d_model, 
        # then the parameter count = 3 * d_model * 8/3 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        return self.w2(silu(self.w1(x)) * self.w3(x))

class SiLUFFN(nn.Module):
    """ FFN with SiLU activation """
    d_model: int
    d_ff: int

    def __init__(self, d_model:int, d_ff:int) -> None:
        super().__init__()
        self.d_model = d_model
        # should be d_ff = 4 * d_model, 
        # then the parameter count = 2 * d_model * 4 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        return self.w2(silu(self.w1(x)))

# uv run pytest -k test_rope
class RotaryPositionalEmbedding(nn.Module):
    """ Rotary Positional Embedding (RoPE) """
    theta: float
    d_k: int
    max_seq_len: int
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer(
            'cos_sin', 
            precompute_freqs_cis(d_k, max_seq_len, theta),
            persistent=False
        )
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # [..., seq_len, dim], [seq_len,] -> [..., seq_len, dim]
        cos_sin = self.cos_sin[:x.size(-2)] if token_positions is None else self.cos_sin[token_positions]
        return apply_rotary_emb(x, cos_sin)


def precompute_freqs_cis(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)) # shape (head_dim/2,)
    t = torch.arange(max_len, device=freqs.device).float() # shape (max_len,)
    freqs = torch.outer(t, freqs) # equal to einsum('i,j->ij', t, freqs), shape (max_len, head_dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, equal to torch.complex(torch.cos(freqs), torch.sin(freqs))
    cos_sin = torch.cat([freqs_cis.real, freqs_cis.imag], dim=-1) # [cos, sin] shape (max_len, head_dim/2 * 2)
    return cos_sin
    
def apply_rotary_emb(x:torch.Tensor, cos_sin:torch.Tensor):
    # x1, x2 = torch.chunk(x, 2, dim=-1)
    x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    x_out = torch.stack([x1 * cos - x2 * sin, 
                         x1 * sin + x2 * cos], dim=-1)
    return x_out.reshape(*x.shape).type_as(x)

# uv run pytest -k test_scaled_dot_product_attention
# uv run pytest -k test_4d_scaled_dot_product_attention
from .nn_utils import softmax
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    D = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# uv run pytest -k test_multihead_self_attention
# pass tests/test_model.py::test_multihead_self_attention
from einops import rearrange
class MultiheadSelfAttention(nn.Module):
    """ Multi-head self-attention layer """
    d_model: int
    num_heads: int
    
    def __init__(self, d_model:int, num_heads:int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fc_qkv = Linear(d_model, 3*d_model)
        self.fc_out = Linear(d_model, d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        seq_len = x.size(1)
        qkv = self.fc_qkv(x)
        qkv = rearrange(qkv, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xq, xk, xv = torch.chunk(qkv, 3, dim=1)
        mask = torch.ones((seq_len, seq_len), device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')
        xo = scaled_dot_product_attention(xq, xk, xv, mask)
        xo = rearrange(xo, 'B nH T Hs -> B T (nH Hs)')
        return self.fc_out(xo)

# uv run pytest -k test_multihead_self_attention
# pass tests/test_model.py::test_multihead_self_attention_with_rope
class MultiheadRoPESelfAttention(nn.Module):
    """ Multi-head self-attention layer with RoPE"""
    d_model: int
    num_heads: int
    
    def __init__(
        self, 
        d_model:int, 
        num_heads:int,
        max_seq_len:int,
        theta:float,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fc_qkv = Linear(d_model, 3*d_model)
        self.fc_out = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)
    
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        seq_len = x.size(1)
        qkv = self.fc_qkv(x)
        qkv = rearrange(qkv, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xq, xk, xv = torch.chunk(qkv, 3, dim=1)
        xq = self.rope(xq, token_positions)
        xk = self.rope(xk, token_positions)
        mask = torch.ones((seq_len, seq_len), device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')
        xo = scaled_dot_product_attention(xq, xk, xv, mask)
        xo = rearrange(xo, 'B nH T Hs -> B T (nH Hs)')
        return self.fc_out(xo)

from functools import lru_cache

@lru_cache(1)
def get_rope(theta: float, d_k: int, max_seq_len: int) -> RotaryPositionalEmbedding:
    return RotaryPositionalEmbedding(theta, d_k, max_seq_len)

class TransformerAttention(nn.Module):
    """ Transformer Attention with RoPE for TransformerBlock """
    d_model: int
    num_heads: int
    
    def __init__(
        self, 
        d_model:int, 
        num_heads:int,
        max_seq_len:int,
        theta:float,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = get_rope(theta, self.head_dim, max_seq_len)
    
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        seq_len = x.size(1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        xq = rearrange(q, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xk = rearrange(k, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xv = rearrange(v, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        if self.rope is not None:
            xq = self.rope(xq, token_positions)
            xk = self.rope(xk, token_positions)
        mask = torch.ones((seq_len, seq_len), device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')
        xo = scaled_dot_product_attention(xq, xk, xv, mask)
        xo = rearrange(xo, 'B nH T Hs -> B T (nH Hs)')
        return self.output_proj(xo)


# uv run pytest -k test_transformer_block
class TransformerBlock(nn.Module):
    """ Transformer Block """
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len:int,
        theta:float,
        ffn_type:str = 'swiglu',
        use_post_norm:bool = False,
        remove_rmsnorm:bool = False,
        remove_rope:bool = False,
    ) -> None:
        super().__init__()
        self.attn = TransformerAttention(
            d_model, num_heads, max_seq_len, theta
        )
        if remove_rope:
            self.attn.rope = None

        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model, d_ff)
        elif ffn_type == 'silu':
            self.ffn = SiLUFFN(d_model, d_ff)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.use_post_norm = use_post_norm
        if remove_rmsnorm:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        if self.use_post_norm:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x

# uv run pytest -k test_transformer_lm
class TransformerLM(nn.Module):
    """ Transformer Language Model """
    def __init__(
        self,
        vocab_size:int,
        context_length:int,
        d_model:int,
        num_layers:int,
        num_heads:int,
        d_ff:int,
        rope_theta:float,
        ffn_type:str = 'swiglu',
        use_post_norm:bool = False,
        remove_rmsnorm:bool = False,
        remove_rope:bool = False,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta,
                ffn_type=ffn_type,
                use_post_norm=use_post_norm,
                remove_rmsnorm=remove_rmsnorm,
                remove_rope=remove_rope,
            )
            for _ in range(num_layers)
        ])
        if remove_rmsnorm:
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.max_seq_len = context_length
        self.apply(init_weights)
        
    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        # [batch, seq_len] -> [batch, seq_len, vocab_size]
        seq_len = token_ids.size(1)
        assert seq_len <= self.max_seq_len, "Sequence length exceeds model capacity"
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
