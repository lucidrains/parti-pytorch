import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)

# classes

class Parti(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
