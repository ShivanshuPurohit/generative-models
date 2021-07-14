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