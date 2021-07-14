import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from jax import jit
import numpy as np
from functools import partial
from typing import Any
import h5py


def minibatch_stddev_layer(x, group_size=None, num_new_features=1):
    if group_size is not None:
        group_size = x.shape[0]
    else:
        group_size = min(group_size, x.shape[0])
    G = group_size
    F = num_new_features
    _, H, W, C = x.shape
    c = C // F
    # [N, H, W, C] cast to fp32
    y = x.astype(jnp.float32)
    # [G, m, H, W, F, c] Split minibatch N into n groups of size G, and channels C into F groups of size c
    y = jnp.reshape(y, (G, -1, H, W, F, c))
    # subtract mean over group
    y -= jnp.mean(y, axis=0)
    # calculate variance over group
    y = jnp.mean(jnp.square(y), axis=0)
    # calculate stddev over group
    y = jnp.sqrt(y + 1e-8)
    # [n, F] take average over channels and pixels
    y = jnp.mean(y, axis=(1, 2, 4))
    # [n, F] cast back to original data type
    y = y.astype(x.dtype)
    # [N, 1, 1, F] add missing dims
    y = jnp.reshape(y, newshape=(-1, 1, 1, F))
    # [N, H, W, C] replicate over groups and pixels
    y = jnp.tile(y, (G, H, W, 1))
    return jnp.concatenate((x, y), axis=3)


def apply_activation(x, activation='Linear', alpha=0.2, gain=np.sqrt(2)):
    if activation == 'relu':
        return jax.nn.relu(x) * gain
    elif activation == 'leaky_relu':
        return jax.nn.leaky_relu(x, negative_slope=alpha) * gain
    return x


def get_weight(shape, lr_multiplier=1, bias=True, param_dict=None, layer_name='',
               key=None, dtype='float32'):
    if param_dict is None:
        w = random.normal(key, shape=shape, dtype=dtype) / lr_multiplier
        if bias: b = jnp.zeros(shape=(shape[-1],), dtype=dtype)
    else:
        w = jnp.array(param_dict[layer_name]['weight']).astype(dtype)
    if bias: return w, b
    return w


def equalize_lr_weights(w, lr_multiplier=1):
    """
    Equalize learning rates for weights in a layer.
    Args: 
        w: weights. shape: [kernel, kernel, fmaps_in, fmaps_out]
        lr_multiplier: learning rate multiplier
    Returns:
        scaled weights
    """
    in_features = np.prod(w.shape[-1])
    gain = lr_multiplier / np.sqrt(in_features)
    w *= gain
    return w


def eqalize_lr_bias(b, lr_multiplier=1):
    """
    Equalize learning rates for bias in a layer.
    Args: 
        b: bias. shape: [fmaps_out]
        lr_multiplier: learning rate multiplier
    Returns:
        scaled bias
    """
    gain = lr_multiplier
    b *= gain
    return b


def normalize_2nd_momment(x, eps=1e-8):
    return x * jax.lax.rsqrt(jax.mean(jnp.square(x), axis=1, keepdims=True) + eps)


def setup_filter(f, normalize=True, flip_filter=False, gain=1, separable=None):
    """
    Convenience function to setup a filter for convolution.
    Args:
        f: filter kernel. shape: [kernel, kernel, in_channels, out_channels]
        normalize: whether to normalize the filter
        flip_filter: whether to flip the filter
        gain: gain to scale the filter by
        separable: whether to use separable filters
    Returns:
        filter: normalized and flipped filter. Shape: [filter_height, filter_width] or [filter_taps]
    """
    if f is None:
        f = 1
    f = jnp.array(f, dtype=jnp.float32)
    assert f.ndim in [0, 1, 2]
    assert f.size > 0
    if f.ndim == 0:
        f = f[jnp.newaxis]

    if separable is None:
        separable = (f.ndim == 1 and f.size >= 0)
    if f.ndim == 1 and not separable:
        f = jnp.outer(f, f)
    assert f.ndim == (1 if separable else 2)

    if normalize:
        f /= jnp.sum(f)
    if flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)
    f = f * (gain ** (f.ndim / 2))
    return f