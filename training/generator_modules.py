import torch
import torch.nn as nn

import numpy as np
from torch_utils.ops import bias_act
import torch.nn.functional as F
from einops import rearrange, pack, unpack, repeat, reduce
from functools import partial
from torch import nn, einsum


def exists(val):
    return val is not None


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps, max=1e32))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        return normed * self.scale * self.gamma


class TextAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        mask_self_value = -1e2
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.mask_self_value = mask_self_value

        self.norm = RMSNorm(dim)
        self.to_qk = nn.Linear(dim, dim_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_inner, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, encodings, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch, device = encodings.shape[0], encodings.device

        encodings = self.norm(encodings)

        h = self.heads

        qk, v = self.to_qk(encodings), self.to_v(encodings)
        qk, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (qk, v))

        q, k = qk, qk

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # l2 distance

        sim = -torch.cdist(q, k, p = 2) * self.scale

        # following what was done in reformer for shared query / key space
        # omit attention to self

        self_mask = torch.eye(sim.shape[-2], device = device, dtype = torch.bool)
        self_mask = F.pad(self_mask, (1, 0), value = False)

        sim = sim.masked_fill(self_mask, self.mask_self_value)

        # key padding mask

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = repeat(mask, 'b n -> (b h) 1 n', h = h)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out)

# feedforward
def FeedForward(
    dim,
    mult = 4,
    channel_first = False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),
        proj(dim, dim_hidden),
        nn.GELU(),
        proj(dim_hidden, dim)
    )

class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TextAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)


class EoTTransfer(torch.nn.Module):
    def __init__(self, in_dim, dim=192, depth=4, eot_channel=None):
        super().__init__()
        self.eot_channel = eot_channel
        self.project_in = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = int(dim/8),
            heads = 8
        )
        self.project_out = torch.nn.Sequential(
            FullyConnectedLayer(77*dim, 16 * eot_channel, lr_multiplier=1.0)
            # torch.nn.utils.spectral_norm(FullyConnectedLayer(in_dim, 16 * eot_channel, lr_multiplier=1.0)),
            # torch.nn.SiLU(),
        )

    def forward(self, eot):
        x = self.project_in(eot)
        x = self.transformer(x)
        x = self.project_out(x.reshape(x.shape[0], -1))
        x = x.reshape(x.shape[0], self.eot_channel, 4, 4)
        return x.to(torch.float32)


class Emb2EotTransfer(torch.nn.Module):
    def __init__(self, in_dim, dim=192, depth=4, out_channel=None):
        super().__init__()
        self.project_in = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=int(dim / 8),
            heads=8
        )
        self.project_out = torch.nn.Sequential(
            FullyConnectedLayer(77*dim, out_channel, lr_multiplier=1.0)
            # torch.nn.utils.spectral_norm(FullyConnectedLayer(in_dim, out_channel, lr_multiplier=1.0)),
            # torch.nn.SiLU(),
        )

    def forward(self, emb):
        x = self.project_in(emb)
        x = self.transformer(x)
        x = self.project_out(x.reshape(x.shape[0], -1))
        return x.to(torch.float32)