import torch
import torch.nn as nn
import math
from einops import rearrange

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

def norm(x:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """Functional RMS normalization without learnable parameters"""
    var = x.pow(2).mean(dim=-1, keepdim=True) + eps
    return x * var.rsqrt()

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

class VQFFN(nn.Module):
    """ FFN with softmax """
    d_model: int
    d_ff: int

    def __init__(self, d_model:int, d_ff:int) -> None:
        super().__init__()
        self.d_model = d_model
        # should be d_ff = 4 * d_model, 
        # then the parameter count = 2 * d_model * 4 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.codebook = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        probs = torch.softmax(self.codebook(x), dim=-1)
        return torch.einsum('...k,kd->...d', probs, self.codebook.weight)

class MHVQFFN(nn.Module):
    """ Multi-head vq FFN with softmax """
    d_model: int
    codebook_size: int
    code_dim: int

    def __init__(self, d_model:int, codebook_size:int, num_codebook:int=4, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        self.code_dim = d_model // num_codebook
        self.codebook_size = codebook_size
        self.weight = nn.Parameter(
            torch.empty((num_codebook, codebook_size, self.code_dim), **kwargs)
        )
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.codebook_size + self.code_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=self.code_dim)
        scores = torch.einsum('...htd,hcd->...htc', x, self.weight)
        probs = torch.softmax(scores, dim=-1)
        x = torch.einsum('...htc,hcd->...htd', probs, self.weight)
        x = rearrange(x, 'B nH T Hs -> B T (nH Hs)')
        return x


class VQFFN1(nn.Module):
    """ FFN with softmax """
    d_model: int
    d_ff: int

    def __init__(self, d_model:int, d_ff:int, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        # should be d_ff = 4 * d_model, 
        # then the parameter count = 2 * d_model * 4 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.weight = nn.Parameter(
            torch.empty((d_ff, d_model), **kwargs)
        )
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.d_ff + self.d_model))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        scores = torch.einsum('...d,cd->...c', norm(x), norm(self.weight))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum('...c,cd->...d', probs, self.weight)

class MHVQFFN1(nn.Module):
    """ Multi-head vq FFN with softmax """
    d_model: int
    codebook_size: int
    code_dim: int

    def __init__(self, d_model:int, codebook_size:int, num_codebook:int=4, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        self.code_dim = d_model // num_codebook
        self.codebook_size = codebook_size
        self.weight = nn.Parameter(
            torch.empty((num_codebook, codebook_size, self.code_dim), **kwargs)
        )
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.codebook_size + self.code_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=self.code_dim)
        scores = torch.einsum('...htd,hcd->...htc', norm(x), norm(self.weight))
        probs = torch.softmax(scores, dim=-1)
        x = torch.einsum('...htc,hcd->...htd', probs, self.weight)
        x = rearrange(x, 'B nH T Hs -> B T (nH Hs)')
        return x


class VQSiLUFFN(nn.Module):
    """ FFN with softmax """
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
        probs = torch.softmax(self.w1(x), dim=-1)
        return self.w2(probs)


class MHVQSiLUFFN(nn.Module):
    """ Multi-head vq FFN with softmax """
    d_model: int
    codebook_size: int
    code_dim: int

    def __init__(self, d_model:int, codebook_size:int, num_codebook:int=4, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        self.code_dim = d_model // num_codebook
        self.codebook_size = codebook_size
        self.weight1 = nn.Parameter(
            torch.empty((num_codebook, codebook_size, self.code_dim), **kwargs)
        )
        self.weight2 = nn.Parameter(
            torch.empty((num_codebook, codebook_size, self.code_dim), **kwargs)
        )
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.codebook_size + self.code_dim))
        for weight in [self.weight1, self.weight2]:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=self.code_dim)
        scores = torch.einsum('...htd,hcd->...htc', x, self.weight1)
        probs = torch.softmax(scores, dim=-1)
        x = torch.einsum('...htc,hcd->...htd', probs, self.weight2)
        x = rearrange(x, 'B nH T Hs -> B T (nH Hs)')
        return x

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

class KVCache(nn.Module):
    def __init__(self, batch_size, n_kv_heads, seq_length, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, n_kv_heads, seq_length, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, positions, xk, xv):
        seq_len = xk.size(2)
        start_pos = positions[0].item()  # assuming all positions are the same in the batch
        self.cache_k[:, :, positions] = xk
        self.cache_v[:, :, positions] = xv        
        cached_k = self.cache_k[:, :, :start_pos + seq_len]
        cached_v = self.cache_v[:, :, :start_pos + seq_len]
        return cached_k, cached_v

class TransformerAttention(nn.Module):
    """ Transformer Attention with RoPE for TransformerBlock """
    d_model: int
    num_heads: int
    theta: float
    
    def __init__(
        self, 
        d_model:int, 
        num_heads:int,
        max_seq_len:int,
        theta:float,
        **kwargs
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.theta = theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # Handle remove_rope option
        if not kwargs.get('remove_rope', False):
            self.rope = get_rope(theta, self.head_dim, max_seq_len)
        else:
            self.rope = None

        # Handle add_qknorm option
        self.add_qknorm = kwargs.get('add_qknorm', False)
        if self.add_qknorm:
            print("Using QK normalization in TransformerAttention")
        
        # KV cache will be managed by inference context manager
        self.cache = None
    
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        seq_len = x.size(1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        xq = rearrange(q, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xk = rearrange(k, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        xv = rearrange(v, 'B T (nH Hs) -> B nH T Hs', Hs=self.head_dim)
        
        if self.add_qknorm:
            xq = norm(xq)
            xk = norm(xk)
        
        if self.rope is not None:
            xq = self.rope(xq, token_positions)
            xk = self.rope(xk, token_positions)
        
        # KV cache update
        if self.cache is not None and token_positions is not None:
            xk, xv = self.cache.update(token_positions, xk, xv)
            cache_len = token_positions[0].item()  # assuming all positions are the same in the batch
            mask = torch.hstack(
                [torch.ones((seq_len, cache_len), device=x.device, dtype=x.dtype),
                 torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype).tril()]
            )
            mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')
        else:
            # Standard causal mask for non-cached case
            mask = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype).tril()
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
        **kwargs
    ) -> None:
        super().__init__()
        
        # Extract options from kwargs with defaults
        ffn_type = kwargs.get('ffn_type', 'swiglu')
        use_post_norm = kwargs.get('use_post_norm', False)
        remove_rmsnorm = kwargs.get('remove_rmsnorm', False)
        
        # Create attention layer with kwargs
        self.attn = TransformerAttention(
            d_model, num_heads, max_seq_len, theta, **kwargs
        )

        # Create FFN layer
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model, d_ff)
        elif ffn_type == 'silu':
            self.ffn = SiLUFFN(d_model, d_ff)
        elif ffn_type == 'vq':
            self.ffn = VQFFN(d_model, d_ff)
        elif ffn_type == 'mhvq':
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHVQFFN(d_model, codebook_size, num_codebook)
        elif ffn_type == 'vq1':
            self.ffn = VQFFN1(d_model, d_ff)
        elif ffn_type == 'mhvq1':
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHVQFFN1(d_model, codebook_size, num_codebook)
        elif ffn_type == 'vqsilu':
            self.ffn = VQSiLUFFN(d_model, d_ff)
        elif ffn_type == 'mhvqsilu':
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHVQSiLUFFN(d_model, codebook_size, num_codebook)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        # Create normalization layers
        self.use_post_norm = use_post_norm
        if remove_rmsnorm:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        if self.use_post_norm:
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(self.ln1(x), token_positions)
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
        **kwargs
    ) -> None:
        super().__init__()
        remove_rmsnorm = kwargs.get('remove_rmsnorm', False)
        tie_embeddings = kwargs.get('tie_embeddings', False)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, **kwargs
            ) for _ in range(num_layers)
        ])
        if remove_rmsnorm:
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.max_seq_len = context_length
        self.apply(init_weights)
        
    def forward(self, token_ids:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # [batch, seq_len] -> [batch, seq_len, vocab_size]
        seq_len = token_ids.size(1)
        assert seq_len <= self.max_seq_len, "Sequence length exceeds model capacity"
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
