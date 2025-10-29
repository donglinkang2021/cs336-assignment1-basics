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

def outer_product(mat: torch.Tensor, code_dims: list[int], split_dim: int = 0) -> torch.Tensor:
    """
    Compute outer product: chunk first, then compute outer product
    Args:
        mat: Tensor of shape (..., d) where one dimension will be split
        code_dims: Sizes of each chunk
        split_dim: Which dimension to split (0 or 1)
    Returns:
        Tensor of shape (prod(code_dims), d) if split_dim=0
        Tensor of shape (n, prod(code_dims)) if split_dim=1
    """
    # Split the matrix according to code_dims
    mats = mat.split(code_dims, dim=split_dim)
    
    if len(mats) == 1:
        return mats[0]
    
    if split_dim == 0:
        # Original behavior: split first dimension
        # Build einsum string, e.g., "az,bz,cz->abcz"
        indices = [chr(ord('a') + i) for i in range(len(mats))]
        einsum_str = ','.join(f'{idx}z' for idx in indices) + '->' + ''.join(indices) + 'z'
        result = torch.einsum(einsum_str, *mats)
        return result.reshape(-1, mat.shape[-1])
    else:
        # Split second dimension
        # Build einsum string, e.g., "za,zb,zc->zabc"
        indices = [chr(ord('a') + i) for i in range(len(mats))]
        einsum_str = ','.join(f'z{idx}' for idx in indices) + '->' + 'z' + ''.join(indices)
        result = torch.einsum(einsum_str, *mats)
        return result.reshape(mat.shape[0], -1)

class OuterLinear(nn.Module):
    """Dynamic Weights via Outer Product of Multiple Low-Rank Tensors"""
    feat_dim: int
    code_dims: list[int]
    out_features: int
    weight: torch.Tensor
    split_dim: int
    
    def __init__(self, feat_dim: int, code_dims: list[int], split_dim:int = 0, outer_norm:bool=True, **kwargs) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.code_dims = code_dims
        self.split_dim = split_dim
        self.out_features = math.prod(code_dims)
        self.use_norm = outer_norm
        
        # Single weight matrix containing all code chunks
        total_codes = sum(code_dims)
        if split_dim == 0:
            self.weight = nn.Parameter(torch.empty((total_codes, feat_dim), **kwargs))
        else:
            self.weight = nn.Parameter(torch.empty((feat_dim, total_codes), **kwargs))
        self._init_weights()

    def _init_weights(self):
        # according to each chunk
        start = 0
        for code_dim in self.code_dims:
            std = math.sqrt(2.0 / (self.feat_dim + code_dim))
            nn.init.trunc_normal_(
                self.weight[start:start+code_dim] if self.split_dim == 0 
                    else self.weight[:, start:start+code_dim],
                mean=0.0, std=std, a=-3*std, b=3*std
            )
            start += code_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dynamic_weight = outer_product(self.weight, self.code_dims, self.split_dim)
        if self.use_norm:
            return norm(torch.einsum('...i,oi->...o', x, dynamic_weight))
        else:
            return torch.einsum('...i,oi->...o', x, dynamic_weight)

class OuterSiLU(nn.Module):
    d_model: int
    code_dims: list[int]

    def __init__(self, d_model:int, code_dims:list[int], outer_norm:bool) -> None:
        super().__init__()
        self.d_model = d_model
        self.w1 = OuterLinear(d_model, code_dims, 0, outer_norm)
        self.w2 = OuterLinear(d_model, code_dims, 1, outer_norm)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))

def outer_product1(x: torch.Tensor, code_dims: list[int]) -> torch.Tensor:
    """Compute outer product: chunk first, then compute outer product"""
    # Split the matrix according to code_dims
    mats = x.split(code_dims, dim=-1)
    # build einsum like '...a,...b,...c->...abc'
    letters = [chr(ord('a') + i) for i in range(len(mats))]
    einsum_str = ','.join(f'...{l}' for l in letters) + '->' + '...' + ''.join(letters)
    out = torch.einsum(einsum_str, *mats)
    # ensure shape is (..., prod(code_dims))
    return out.reshape(*x.shape[:-1], -1)

class DynaLinear(nn.Module):
    """Dynamic Tokens via Outer Product of Multiple Low-Rank Tensors"""
    feat_dim: int
    code_dims: list[int]
    out_features: int
    weight: torch.Tensor
    
    def __init__(self, feat_dim:int, code_dims:list[int], outer_norm:bool=True, **kwargs) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.code_dims = code_dims
        self.out_features = math.prod(code_dims)
        self.use_norm = outer_norm
        
        # Single weight matrix containing all code chunks
        self.weight = nn.Parameter(torch.empty((feat_dim, sum(code_dims)), **kwargs))
        self._init_weights()

    def _init_weights(self):
        # according to each chunk
        start = 0
        for code_dim in self.code_dims:
            std = math.sqrt(2.0 / (self.feat_dim + code_dim))
            nn.init.trunc_normal_(self.weight[:, start:start+code_dim], mean=0.0, std=std, a=-3*std, b=3*std)
            start += code_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('...i,io->...o', x, self.weight)
        x = outer_product1(x, self.code_dims)
        return norm(x) if self.use_norm else x

class DynaSiLU(nn.Module):
    d_model: int
    d_ff: int
    code_dims: list[int]

    def __init__(self, d_model:int, d_ff:int, code_dims:list[int], outer_norm:bool) -> None:
        assert math.prod(code_dims) == d_ff, "Product of code_dims must equal d_ff"
        super().__init__()
        self.d_model = d_model
        self.w1 = DynaLinear(d_model, code_dims, outer_norm)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))

class DynaSwiGLU(nn.Module):
    """ SwiGLU FFN """
    d_model: int
    d_ff: int
    
    def __init__(self, d_model:int, d_ff:int, code_dims:list[int]) -> None:
        assert math.prod(code_dims) == d_ff, "Product of code_dims must equal d_ff"
        super().__init__()
        self.d_model = d_model
        # should be roughly d_ff = 8/3 * d_model, 
        # then the parameter count = 3 * d_model * 8/3 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=code_dims[0])
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=code_dims[1])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        gc = torch.einsum('...i,...j->...ij', silu(self.w1(x)), self.w3(x)).reshape(*x.shape[:-1], -1)
        return self.w2(gc)

class CacheLinear(nn.Module):
    in_features: int
    out_features: int
    num_codebook: int
    code_dim: int
    down_proj: torch.Tensor
    up_proj: torch.Tensor
    
    def __init__(self, in_features:int, out_features:int, num_codebook: int, code_dim: int, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebook = num_codebook
        self.code_dim = code_dim
        self.down_proj = nn.Parameter(torch.empty((num_codebook, in_features, code_dim), **kwargs))
        self.up_proj = nn.Parameter(torch.empty((num_codebook, out_features, code_dim), **kwargs))
        self._init_weights()

    def _init_weights(self):
        down_std = math.sqrt(2.0 / (self.in_features + self.code_dim))
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=down_std, a=-3*down_std, b=3*down_std)
        up_std = math.sqrt(2.0 / (self.out_features + self.code_dim))
        nn.init.trunc_normal_(self.up_proj, mean=0.0, std=up_std, a=-3*up_std, b=3*up_std)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.einsum('...i,hic->...hc', x, self.down_proj)
        x = torch.einsum('...hc,hoc->...ho', x, self.up_proj)
        return norm(x.prod(dim=-2))

class CacheSiLU(nn.Module):
    d_model: int
    d_ff: int
    num_codebook: int
    code_dim: int

    def __init__(self, d_model:int, d_ff:int, num_codebook:int, code_dim:int, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = CacheLinear(d_model, d_ff, num_codebook, code_dim)
        self.w2 = CacheLinear(d_ff, d_model, num_codebook, code_dim)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))

class Head(nn.Module):
    """ FFN without activation """
    d_model: int
    d_ff: int
    head_dim: int

    def __init__(self, d_model:int, d_ff:int, head_dim:int) -> None:
        super().__init__()
        self.d_model = d_model
        # should be d_ff = 4 * d_model, 
        # then the parameter count = 2 * d_model * 4 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.head_dim = head_dim
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        x = self.w1(x)
        x = rearrange(x, 'B T (nH Hs) -> B T nH Hs', Hs=self.head_dim)
        s = torch.einsum('...id,...jd->...ij', x, x).softmax(dim=-1)
        x = torch.einsum('...ij,...jd->...id', s, x)
        x = rearrange(x, 'B T nH Hs -> B T (nH Hs)')
        return self.w2(x)

class SiLUFFN1(nn.Module):
    """ FFN with SiLU activation """
    d_model: int
    d_ff: int

    def __init__(self, d_model:int, d_ff:int) -> None:
        super().__init__()
        self.d_model = d_model
        # should be d_ff = 4 * d_model, 
        # then the parameter count = 2 * d_model * 4 * d_model = 8 * d_model^2
        self.d_ff = d_ff
        self.fc = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [..., d_model] -> [..., d_model]
        return torch.einsum('...k,kd->...d', silu(self.fc(x)), self.fc.weight)


class MHSiLUFFN1(nn.Module):
    """ FFN with SiLU activation """
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
        x = torch.einsum('...htc,hcd->...htd', silu(scores), self.weight)
        x = rearrange(x, 'B nH T Hs -> B T (nH Hs)')
        return x

class MHSiLUFFN(nn.Module):
    """ FFN with SiLU activation """
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
        x = torch.einsum('...htc,hcd->...htd', silu(scores), self.weight2)
        x = rearrange(x, 'B nH T Hs -> B T (nH Hs)')
        return x

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
        # current result swiglu > silu > silu1 > mhvq > vq
        # add norm to codebook and x: not so important(keep same performance), for ln2 maintain the stability
        # w/o tie_embedding: vqsilu and mhvqsilu, not so important(keep same performance), for just one weight is enough
        
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model, d_ff)
        elif ffn_type == 'silu':
            self.ffn = SiLUFFN(d_model, d_ff)
        elif ffn_type == 'cache_silu':
            num_codebook = kwargs.get('num_codebook', 4)
            code_dim = kwargs.get('code_dim', 4)
            self.ffn = CacheSiLU(d_model, d_ff, num_codebook, code_dim)
        elif ffn_type == 'outer_silu':
            code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
            outer_norm = kwargs.get('outer_norm', True)
            self.ffn = OuterSiLU(d_model, code_dims, outer_norm)
        elif ffn_type == 'dyna_silu':
            code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
            outer_norm = kwargs.get('outer_norm', True)
            self.ffn = DynaSiLU(d_model, d_ff, code_dims, outer_norm)
        elif ffn_type == 'dyna_swiglu':
            code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
            self.ffn = DynaSwiGLU(d_model, d_ff, code_dims)
        elif ffn_type == 'identity':
            # Remove FFN layer - just use identity mapping
            self.ffn = nn.Identity()
        elif ffn_type == 'head':
            head_dim = kwargs.get('head_dim', d_ff // 4)
            self.ffn = Head(d_model, d_ff, head_dim)
        elif ffn_type == 'silu1':
            # tie_embedding version of silu
            self.ffn = SiLUFFN1(d_model, d_ff)
        elif ffn_type == 'mhsilu1':
            # use multihead code to increase codebook_size in a efficient way
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHSiLUFFN1(d_model, codebook_size, num_codebook)
        elif ffn_type == 'mhsilu':
            # use multihead code to increase codebook_size in a efficient way
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHSiLUFFN(d_model, codebook_size, num_codebook)
        elif ffn_type == 'vq':
            # use softmax(x) to replace silu(x)
            self.ffn = VQFFN(d_model, d_ff)
        elif ffn_type == 'mhvq':
            # use multihead code to increase codebook_size in a efficient way
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHVQFFN(d_model, codebook_size, num_codebook)
        elif ffn_type == 'vq1':
            # add norm to codebook and x of vqffn
            self.ffn = VQFFN1(d_model, d_ff)
        elif ffn_type == 'mhvq1':
            # add norm to codebook and x of mhvqffn
            num_codebook = kwargs.get('num_codebook', 4)
            codebook_size = kwargs.get('codebook_size', d_ff)
            self.ffn = MHVQFFN1(d_model, codebook_size, num_codebook)
        elif ffn_type == 'vqsilu':
            # tie_embedding ablation of vqffn
            self.ffn = VQSiLUFFN(d_model, d_ff)
        elif ffn_type == 'mhvqsilu':
            # tie_embedding ablation of mhvqffn
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
