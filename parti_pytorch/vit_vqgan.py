import copy
import math
from math import sqrt
from functools import partial, wraps

from vector_quantize_pytorch import VectorQuantize as VQ

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision

from einops import rearrange, reduce, repeat
from einops_exts import rearrange_many
from einops.layers.torch import Rearrange

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# tensor helper functions

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# fourier

class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        dim,
        height_or_width,
        theta = 10000
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta

        hw_range = torch.arange(height_or_width)
        coors = torch.stack(torch.meshgrid(hw_range, hw_range, indexing = 'ij'), dim = -1)
        coors = rearrange(coors, 'h w c -> h w c')
        self.register_buffer('coors', coors, persistent = False)

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(self.coors, 'h w c -> h w c 1') * rearrange(emb, 'j -> 1 1 1 j')
        fourier = torch.cat((emb.sin(), emb.cos()), dim = -1)
        fourier = repeat(fourier, 'h w c d -> b (c d) h w', b = x.shape[0])
        return torch.cat((x, fourier), dim = 1)

# vqgan vae

class ChanLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim)
        self.activation = leaky_relu()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        groups = 8
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)

# discriminator

class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 8,
        init_kernel_size = 5,
        cross_embed_kernel_sizes = (3, 7, 15)
    ):
        super().__init__()
        init_dim, *_, final_dim = dims
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = MList([nn.Sequential(
            CrossEmbedLayer(channels, cross_embed_kernel_sizes, init_dim, stride = 1),
            leaky_relu()
        )])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                leaky_relu(),
                nn.GroupNorm(groups, dim_out),
                ResnetBlock(dim_out, dim_out),
            ))

        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(final_dim, final_dim, 1),
            leaky_relu(),
            nn.Conv2d(final_dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

# ViT encoder / decoder

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x)

class SPT(nn.Module):
    """ https://arxiv.org/abs/2112.13492 """

    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            ChanLayerNorm(patch_dim),
            nn.Conv2d(patch_dim, dim, 1)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32,
        fmap_size = None,
        rel_pos_bias = False
    ):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.primer_ds_convs = nn.ModuleList([PEG(inner_dim) for _ in range(3)])

        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

        self.rel_pos_bias = None
        if rel_pos_bias:
            assert exists(fmap_size)
            self.rel_pos_bias = RelPosBias2d(fmap_size, heads)

    def forward(self, x):
        fmap_size = x.shape[-1]
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = [ds_conv(t) for ds_conv, t in zip(self.primer_ds_convs, (q, k, v))]
        q, k, v = rearrange_many((q, k, v), 'b (h d) x y -> b h (x y) d', h = h)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(self.rel_pos_bias):
            sim = sim + self.rel_pos_bias(sim)

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = fmap_size, y = fmap_size)
        return self.to_out(out)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1, bias = False),
        nn.GELU(),
        PEG(dim * mult),
        nn.Conv2d(dim * mult, dim, 1, bias = False)
    )

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        fmap_size = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, fmap_size = fmap_size, rel_pos_bias = True),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = ChanLayerNorm(dim)

    def forward(self, x):
        for peg, attn, ff in self.layers:
            x = peg(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTEncDec(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        patch_size = 16,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.encoded_dim = dim
        self.patch_size = patch_size

        input_dim = channels * (patch_size ** 2)
        fmap_size = image_size // patch_size

        self.encoder = nn.Sequential(
            SPT(dim = dim, patch_size = patch_size, channels = channels),
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
        )

        self.decoder = nn.Sequential(
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
            nn.Sequential(
                SinusoidalPosEmb(dim // 2, height_or_width = fmap_size),
                nn.Conv2d(2 * dim, dim * 4, 3, bias = False, padding = 1),
                nn.Tanh(),
                nn.Conv2d(dim * 4, input_dim, 1, bias = False),
            ),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    @property
    def last_dec_layer(self):
        return self.decoder[-2][-1].weight

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

# vit vqgan vae

class VitVQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
        vq_codebook_dim = 64,
        vq_codebook_size = 512,
        vq_decay = 0.9,
        vq_commitment_weight = 1.,
        vq_kmeans_init = True,
        use_vgg_and_gan = True,
        discr_layers = 4,
        **kwargs
    ):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)

        self.image_size = image_size
        self.channels = channels
        self.codebook_size = vq_codebook_size

        self.enc_dec = ViTEncDec(
            dim = dim,
            image_size = image_size,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        self.vq = VQ(
            dim = self.enc_dec.encoded_dim,
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            kmeans_init = vq_kmeans_init,
            accept_image_fmap = True,
            use_cosine_sim = True,
            **vq_kwargs
        )

        # reconstruction loss

        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # turn off GAN and perceptual loss if grayscale

        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # preceptual loss

        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # gan related losses

        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    def codebook(self):
        return self.vq.codebook

    def get_fmap_from_codebook(self, indices):
        codes = self.codebook[indices]
        fmap = self.vq.project_out(codes)
        return rearrange(fmap, 'b h w c -> b c h w')

    def encode(self, fmap, return_indices_and_loss = True):
        fmap = self.enc_dec.encode(fmap)

        fmap, indices, commit_loss = self.vq(fmap)

        if not return_indices_and_loss:
            return fmap

        return fmap, indices, commit_loss

    def decode(self, fmap):
        return self.enc_dec.decode(fmap)

    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        apply_grad_penalty = True
    ):
        batch, channels, height, width, device = *img.shape, img.device
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        fmap, indices, commit_loss = self.encode(img, return_indices_and_loss = True)

        fmap = self.decode(fmap)

        if not return_loss and not return_discr_loss:
            return fmap

        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            fmap.detach_()
            img.requires_grad_()

            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, fmap

            return loss

        # reconstruction loss

        recon_loss = self.recon_loss_fn(fmap, img)

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # perceptual loss

        img_vgg_input = img
        fmap_vgg_input = fmap

        if img.shape[1] == 1:
            # handle grayscale for vgg
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(fmap))

        # calculate adaptive weight

        last_dec_layer = self.enc_dec.last_dec_layer

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, fmap

        return loss
