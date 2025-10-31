import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

def ffn_factory(d_model:int, d_ff:int, ffn_type:str, **kwargs) -> nn.Module:
    ffn = None
    if ffn_type == 'swiglu':
        ffn = SwiGLU(d_model, d_ff)
    elif ffn_type == 'silu':
        ffn = SiLUFFN(d_model, d_ff)
    elif ffn_type == 'cache_silu':
        num_codebook = kwargs.get('num_codebook', 4)
        code_dim = kwargs.get('code_dim', 4)
        ffn = CacheSiLU(d_model, d_ff, num_codebook, code_dim)
    elif ffn_type == 'outer_silu':
        code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
        outer_norm = kwargs.get('outer_norm', True)
        ffn = OuterSiLU(d_model, code_dims, outer_norm)
    elif ffn_type == 'dyna_silu':
        code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
        outer_norm = kwargs.get('outer_norm', True)
        ffn = DynaSiLU(d_model, d_ff, code_dims, outer_norm)
    elif ffn_type == 'dynam_silu':
        code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
        num_codebook = kwargs.get('num_codebook', 4)
        outer_norm = kwargs.get('outer_norm', True)
        ffn = DynamSiLU(d_model, d_ff, code_dims, num_codebook, outer_norm)
    elif ffn_type == 'outerm_silu':
        code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
        num_codebook = kwargs.get('num_codebook', 4)
        outer_norm = kwargs.get('outer_norm', True)
        ffn = OutermSiLU(d_model, d_ff, code_dims, num_codebook, outer_norm)
    elif ffn_type == 'dyna_swiglu':
        code_dims = list(kwargs.get('code_dims', [int(math.sqrt(d_ff))] * 2))
        ffn = DynaSwiGLU(d_model, d_ff, code_dims)
    elif ffn_type == 'identity':
        # Remove FFN layer - just use identity mapping
        ffn = nn.Identity()
    elif ffn_type == 'head':
        head_dim = kwargs.get('head_dim', d_ff // 4)
        ffn = Head(d_model, d_ff, head_dim)
    elif ffn_type == 'silu1':
        # tie_embedding version of silu
        ffn = SiLUFFN1(d_model, d_ff)
    elif ffn_type == 'mhsilu1':
        # use multihead code to increase codebook_size in a efficient way
        num_codebook = kwargs.get('num_codebook', 4)
        codebook_size = kwargs.get('codebook_size', d_ff)
        ffn = MHSiLUFFN1(d_model, codebook_size, num_codebook)
    elif ffn_type == 'mhsilu':
        # use multihead code to increase codebook_size in a efficient way
        num_codebook = kwargs.get('num_codebook', 4)
        codebook_size = kwargs.get('codebook_size', d_ff)
        ffn = MHSiLUFFN(d_model, codebook_size, num_codebook)
    elif ffn_type == 'vq':
        # use softmax(x) to replace silu(x)
        ffn = VQFFN(d_model, d_ff)
    elif ffn_type == 'mhvq':
        # use multihead code to increase codebook_size in a efficient way
        num_codebook = kwargs.get('num_codebook', 4)
        codebook_size = kwargs.get('codebook_size', d_ff)
        ffn = MHVQFFN(d_model, codebook_size, num_codebook)
    elif ffn_type == 'vq1':
        # add norm to codebook and x of vqffn
        ffn = VQFFN1(d_model, d_ff)
    elif ffn_type == 'mhvq1':
        # add norm to codebook and x of mhvqffn
        num_codebook = kwargs.get('num_codebook', 4)
        codebook_size = kwargs.get('codebook_size', d_ff)
        ffn = MHVQFFN1(d_model, codebook_size, num_codebook)
    elif ffn_type == 'vqsilu':
        # tie_embedding ablation of vqffn
        ffn = VQSiLUFFN(d_model, d_ff)
    elif ffn_type == 'mhvqsilu':
        # tie_embedding ablation of mhvqffn
        num_codebook = kwargs.get('num_codebook', 4)
        codebook_size = kwargs.get('codebook_size', d_ff)
        ffn = MHVQSiLUFFN(d_model, codebook_size, num_codebook)
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")
    return ffn


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

def norm(x:torch.Tensor, eps:float=1e-12) -> torch.Tensor:
    """Functional RMS normalization without learnable parameters"""
    var = x.pow(2).mean(dim=-1, keepdim=True) + eps
    return x * var.rsqrt()

# uv run pytest -k test_silu
def silu(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

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

class DynamLinear(nn.Module):
    """Dynamic Tokens via Outer Product of Multiple Low-Rank Tensors"""
    feat_dim: int
    code_dims: list[int]
    out_features: int
    num_codebook: int
    weight: torch.Tensor
    
    def __init__(
            self, 
            feat_dim:int, 
            code_dims:list[int], 
            num_codebook:int, 
            outer_norm:bool=True, 
            **kwargs
        ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.code_dims = code_dims
        self.out_features = math.prod(code_dims)
        self.num_codebook = num_codebook
        self.use_norm = outer_norm
        
        # Single weight matrix containing all code chunks
        self.weight = nn.Parameter(torch.empty((num_codebook, feat_dim, sum(code_dims)), **kwargs))
        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.feat_dim + sum(self.code_dims)))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = repeat(x, '... d -> ... h d', h=self.num_codebook)
        x = torch.einsum('...hi,hio->...ho', x, self.weight)
        x = outer_product1(x, self.code_dims).mean(dim=-2) * math.sqrt(self.num_codebook)
        return norm(x) if self.use_norm else x

class OutermLinear(nn.Module):
    """Dynamic Tokens via Outer Product of Multiple Low-Rank Tensors"""
    feat_dim: int
    code_dims: list[int]
    out_features: int
    num_codebook: int
    weight: torch.Tensor
    
    def __init__(
            self, 
            feat_dim:int, 
            code_dims:list[int], 
            num_codebook:int, 
            outer_norm:bool=True, 
            **kwargs
        ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.code_dims = code_dims
        self.out_features = math.prod(code_dims)
        self.num_codebook = num_codebook
        self.use_norm = outer_norm
        
        # Single weight matrix containing all code chunks
        self.weight = nn.Parameter(torch.empty((num_codebook, feat_dim, sum(code_dims)), **kwargs))
        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.feat_dim + sum(self.code_dims)))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dynamic_weight = outer_product1(self.weight, self.code_dims).mean(0) * math.sqrt(self.num_codebook)
        out = torch.einsum('...i,io->...o', x, dynamic_weight)
        return norm(out) if self.use_norm else out

class DynamSiLU(nn.Module):
    d_model: int
    d_ff: int
    code_dims: list[int]

    def __init__(self, d_model:int, d_ff:int, code_dims:list[int], num_codebook:int, outer_norm:bool) -> None:
        assert math.prod(code_dims) == d_ff, "Product of code_dims must equal d_ff"
        super().__init__()
        self.d_model = d_model
        self.w1 = DynamLinear(d_model, code_dims, num_codebook, outer_norm)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))

class OutermSiLU(nn.Module):
    d_model: int
    d_ff: int
    code_dims: list[int]

    def __init__(self, d_model:int, d_ff:int, code_dims:list[int], num_codebook:int, outer_norm:bool) -> None:
        assert math.prod(code_dims) == d_ff, "Product of code_dims must equal d_ff"
        super().__init__()
        self.d_model = d_model
        self.w1 = OutermLinear(d_model, code_dims, num_codebook, outer_norm)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
