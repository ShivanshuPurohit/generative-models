from functools import partial
import jax 
import jax.numpy as jnp
import flax
from flax import linen as nn
from jax import random
from jax import image
from flax.core import freeze, unfreeze
from einops import repeat
import hps
identity = lambda x: x

def gaussian_kl(mu1, mu2, logsigma1, logsigma2):
    # KL(q(z|x)||p(z))
    return (-0.5 + logsigma2 - logsigma1
            +0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
            / (jnp.exp(logsigma2) ** 2))


def gaussian_sample(mu, sigma, rng):
    # Sample from a normal distribution
    return mu + sigma * random.normal(rng, mu.shape)


Conv1x1 = partial(nn.Conv, kernel_size=1, padding=0, strides=1)
Conv3x3 = partial(nn.Conv, kernel_size=3, padding='SAME', strides=1)


def resize(img, shape):
    n, _, _, c = img.shape      # _ are the batch dims
    # resize according to the batch dims
    return image.resize(img, (n,)+shape+(c,), 'nearest')


def recon_loss(px_z, x):
    return jnp.abs(px_z - x).mean()


def sample(px_z):
    # clip and rescale to [0,255]
    return jnp.round((jnp.clip(px_z, -1, 1) + 1) * 127.5).astype(jnp.uint8)


class Attention(nn.Module):
    H: hps.HParams
    
    def setup(self):
        H = self.H
        self.attention = nn.SelfAttention(n_heads=H.attention_heads, dtype=H.dtype)

    def __call__(self, x):
        res = x
        x = self.attention(normalize(x, self.H)) * np.sqrt(1/x.shape[-1])


def parse_layer_strings(s):
    """
    Parse a string of layer names and return a list of layer names
    """
    layers = []
    for ss in ss.split(','):
        if 'x' in ss:
            res, count = ss.split('x')
            layers.extend([res] * int(count))
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
    # pad the channels with zeros
    return jnp.pad(t, (t.ndim-1)*[(0, 0)] + [(0, new_width - t.shape[-1])], mode='constant')
