import jax.numpy as jnp
from jax import random
from jax.util import safe_map
from flax import linen as nn
from einops import repeat, rearrange

from functools import partial
import itertools
from typing import Optional, Any
import numpy as np

import hps
map = safe_map

# first some helper functions
identity = lambda x: x

def gaussian_kl(mu1, mu2, logsigma1, logsigma2):
    return (-0.5 + logsigma2 - logsigma1
            + 0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
            / (jnp.exp(logsigma2) ** 2))

def gaussian_sample(mu, sigma, rng):
    return mu + sigma * random.normal(rng, mu.shape)

Conv1x1 = partial(nn.Conv, kernel_size=(1, 1), strides=(1, 1))

Conv3x3 = partial(nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

def resize(img, shape):
    n, _, _, c = img.shape
    return image.resize(img, (n,) + shape + (c,), 'nearest')

def recon_loss(px_z, x):
    return jnp.abs(px_z - x).mean()

def sample(px_z):
    return jnp.round((jnp.clip(px_z, -1, 1) + 1) * 127.5).astype(jnp.uint8)

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, count = ss.split('x')
            layers.extend(int(count) * [(int(res), None)])
        elif 'm' in ss:
            res, mixin = ss.split('m')
            layers.append((int(res), int(mixin)))
        elif 'd' in ss:
            res, down_rate = ss.split('d')
            layers.append((int(res), int(down_rate)))
        else:
            res = int(ss)
            layers.append((res, 1))
    return layers

def pad_channels(t, new_width):
    return jnp.pad(t, (t.ndim - 1) * [(0, 0)] + [(0, new_width - t.shape[-1])])

def get_width_settings(s):
    mapping = {}
    if s:
        for ss in s.split(','):
            k, v = ss.split(':')
            mapping[k] = int(v)
    return mapping

def normalize(x, type=None, train=False):
    if type == 'group':
        return nn.GroupNorm()(x)
    elif type == 'batch':
        return nn.BatchNorm(use_running_average=not train, axis_name='batch')(x)
    else:
        return x

def checkpoint(fun, H, args):
    if H.checkpoint:
        return jax.checkpoint(fun)(*args)
    else:
        return fun(*args)

def astype(x, H):
    if H.dtype == 'bfloat16':
        return x.astype(jnp.bfloat16)
    elif H.dtype == 'float32':
        return x.astype(jnp.float32)
    else:
        raise NotImplementedError
        
def has_attn(res_, H):
    attn_res = [int(res) for res in H.attn_res.split(',') if len(res) > 0]
    return res_ in attn_res

# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(
        scale, 'fan_in', 'truncated_normal')
  
  
class Attention(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        self.attention = nn.SelfAttention(num_heads=H.num_heads, dtype=H.dtype)

    def __call__(self, x):
        res = x
        x = self.attention(normalize(x, self.H)) * np.sqrt(1 / x.shape[-1])
        return x + res

      
class Block(nn.Module):
    H: hps.Hyperparams
    middle_width: int
    out_width: int
    down_rate: int = 1
    residual: bool = False
    use_3x3: bool = True
    last_scale: bool = 1.
    up: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        H = self.H
        residual = self.residual
        Conv1x1_ = partial(Conv1x1, dtype=H.dtype)
        Conv3x3_ = partial(Conv3x3 if self.use_3x3 else Conv1x1, dtype=H.dtype)
        if H.block_type == 'bottleneck':
            x_ = Conv1x1_(self.middle_width)(nn.gelu(x))
            x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
            x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
            x_ = Conv1x1_(
                self.out_width, kernel_init=lecun_normal(self.last_scale))(
                    nn.gelu(x_))
        elif H.block_type == 'diffusion':
            middle_width = int(self.middle_width / H.bottleneck_multiple)
            x_ = Conv3x3_(middle_width)(nn.gelu(x))
            x_ = Conv3x3_(
                self.out_width, kernel_init=lecun_normal(self.last_scale))(
                    nn.gelu(x_))
                
        out = x + x_ if residual else x_
        if self.down_rate > 1:
            if self.up:
                out = repeat(out, 'b h w c -> b (h x) (w y) c', x=self.down_rate, y=self.down_rate)
            else:
                window_shape = 2 * (self.down_rate,)
                out = nn.avg_pool(out, window_shape, window_shape)
        return out

    
class EncBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    width: int
    down_rate: int = 1
    use_3x3: bool = True
    last_scale: bool = 1.
    up: bool = False

    def setup(self):
        H = self.H
        width, use_3x3 = self.width, self.use_3x3        
        middle_width = int(width * H.bottleneck_multiple)
        self.pre_layer = Attention(H) if has_attn(self.res, H) else identity
        self.block1 = Block(H, middle_width, width, self.down_rate or 1, True, use_3x3, up=self.up)
        
    def __call__(self, x, train=True):
        return self.block1(self.pre_layer(x), train=train)

class Encoder(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        assert widths[str(int(x.shape[1]))] == H.width
        x = Conv3x3(H.width, dtype=H.dtype)(x)
        blocks = parse_layer_string(H.enc_blocks)
        n_blocks = len(blocks)
        activations = {}
        activations[x.shape[1]] = x  # Spatial dimension
        for res, down_rate in blocks:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            width = widths.get(str(res), H.width)
            block = EncBlock(H, res, width, down_rate or 1, use_3x3, last_scale=np.sqrt(1 / n_blocks))
            x = checkpoint(block.__call__, H, (x,))
            new_res = x.shape[1]
            new_width = widths.get(str(new_res), H.width)
            x = x if (x.shape[3] == new_width) else pad_channels(x, new_width)
            activations[new_res] = x
        return activations


class DecBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    mixin: Optional[int]
    n_blocks: int
    bias_cond: bool

    def setup(self):
        H = self.H
        width = self.width = get_width_settings(
            H.custom_width_str).get(str(self.res), H.width)
        use_3x3 = self.res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim if H.zdim != -1 else width
        self.enc   = Block(H, cond_width, self.zdim * 2,
                           use_3x3=use_3x3)
        self.prior = Block(H, cond_width, self.zdim * 2 + width,
                           use_3x3=use_3x3, last_scale=0.)
        self.z_proj = Conv1x1(
            width, kernel_init=lecun_normal(np.sqrt(1 / self.n_blocks)),
            dtype=self.H.dtype)
        self.resnet = Block(H, cond_width, width, residual=True,
                            use_3x3=use_3x3,
                            last_scale=np.sqrt(1 / self.n_blocks))
        self.z_fn = lambda x: self.z_proj(x)
        self.pre_layer = Attention(H) if has_attn(self.res, H) else identity
        self.bias_x = None
        self.expand_cond = self.mixin is not None
        if self.bias_cond:
            self.bias_x = self.param('bias_'+str(self.res), lambda key, shape: jnp.zeros(shape, dtype=H.dtype), (self.res, self.res, width))

    def sample(self, x, acts, rng):
        x = jnp.broadcast_to(x, acts.shape)
        qm, qv = jnp.split(self.enc(jnp.concatenate([x, acts], 3)), 2, 3)
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        z = gaussian_sample(qm, jnp.exp(qv), rng)
        kl = gaussian_kl(qm, pm, qv, pv)
        #print('sample', jnp.isnan(kl.mean()), jnp.isfinite(kl.mean()))
        return z, x + xpp, kl

    def sample_uncond(self, x, rng, t=None, lvs=None):
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        return (gaussian_sample(pm, jnp.exp(pv) * (t or 1), rng)
                if lvs is None else lvs, x + xpp)

    def add_bias(self, x, batch):
        bias = repeat(self.bias_x, '... -> b ...', b=batch)
        if x is None:
            return bias
        else:
            return x + bias 
    
    def forward(self, acts, rng, x=None):
        if self.expand_cond:
            # Assume width increases monotonically with depth
            x = resize(x[..., :acts.shape[3]], (self.res, self.res))
        if self.bias_cond:
            x = self.add_bias(x, acts.shape[0])
        x = self.pre_layer(x)
        #print('call', jnp.isnan(x.mean()), jnp.isfinite(x.mean()))
        z, x, kl = self.sample(x, acts, rng)
        x = self.resnet(x + self.z_fn(z))
        return z, x, kl
    
    def __call__(self, acts, rng, x=None):
        z, x, kl = self.forward(acts, rng, x=x)
        return x, dict(kl=kl)
    
    def get_latents(self, acts, rng, x=None):
        z, x, kl = self.forward(acts, rng, x=x)
        return x, dict(kl=kl, z=z)
    
    def forward_uncond(self, rng, n, t=None, lvs=None, x=None):
        if self.expand_cond:
            # Assume width increases monotonically with depth
            x = resize(x[..., :self.width], (self.res, self.res))
        if self.bias_cond:
            x = self.add_bias(x, n)
        x = self.pre_layer(x)
        z, x = self.sample_uncond(x, rng, t, lvs)
        return self.resnet(x + self.z_fn(z))

class Decoder(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            bias_cond = (mixin is not None and res <= H.no_bias_above) or (idx == 0)
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks), bias_cond=bias_cond))
            resos.add(res)
        self.dec_blocks = dec_blocks        
        self.gain = self.param('gain', lambda key, shape: jnp.ones(shape, dtype=self.H.dtype), (H.width,))
        self.bias = self.param('bias', lambda key, shape: jnp.zeros(shape, dtype=self.H.dtype), (H.width,))
        self.out_conv = Conv1x1(H.n_channels, dtype=self.H.dtype)
        self.final_fn = lambda x: self.out_conv(x * self.gain + self.bias)

    def __call__(self, activations, rng, get_latents=False):
        stats = []
        for idx, block in enumerate(self.dec_blocks):
            rng, block_rng = random.split(rng)
            acts = activations[block.res]
            f = block.__call__ if not get_latents else block.get_latents
            if idx == 0:
                x, block_stats = checkpoint(f, self.H, (acts, block_rng))
            else:
                x, block_stats = checkpoint(f, self.H, (acts, block_rng, x))
            stats.append(block_stats)
        return self.final_fn(x), stats

    def forward_uncond(self, n, rng, t=None):
        x = None
        for idx, block in enumerate(self.dec_blocks):
            t_block = t[idx] if isinstance(t, list) else t
            rng, block_rng = random.split(rng)
            x = block.forward_uncond(block_rng, n, t_block, x=x)
        return self.final_fn(x)

    def forward_manual_latents(self, n, latents, rng, t=None):
        x = None
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
            rng, block_rng = random.split(rng)
            x = block.forward_uncond(block_rng, n, t, lvs, x=x)
        return self.final_fn(x)

class VDVAE(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def __call__(self, x, rng=None, **kwargs):
        x_target = jnp.array(x) # is this clone?
        rng, uncond_rng = random.split(rng)
        px_z, stats = self.decoder(self.encoder(x), rng)
        ndims = np.prod(x.shape[1:])
        kl = sum((s['kl']/ ndims).sum((1, 2, 3)).mean() for s in stats)
        loss = recon_loss(px_z, x_target)
        return dict(loss=loss + kl, recon_loss=loss, kl=kl), None
        
    def forward_get_latents(self, x, rng):
        return self.decoder(self.encoder(x), rng, get_latents=True)[-1]

    def forward_uncond_samples(self, size, rng, t=None):
        return sample(self.decoder.forward_uncond(size, rng, t=t))

    def forward_samples_set_latents(self, size, latents, rng, t=None):
        return sample(self.decoder.forward_manual_latents(size, latents, rng, t=t))
