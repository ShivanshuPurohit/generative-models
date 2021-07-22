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


def upfirdn2d(x, f, padding=(2, 1, 2, 1), up=1, down=1, strides=(1, 1), flip_filter=False, gain=1):
    
    if f is None:
        f = jnp.ones((1, 1))
    B, H, W, C = x.shape
    padx0, padx1, pady0, pady1 = padding

    #upsample by inserting zeros
    x = jnp.reshape(x, (B, H, 1, W, 1, C))
    x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, up-1), (0, 0), (0, up-1), (0, 0)))
    x = jnp.reshape(x, newshape=(B, H*up, W*up, C))

    #padding
    x = jnp.pad(x, pad_width=((0, 0), (max(pady0, 0), max(pady1, 0)), (max(padx0, 0), max(padx1, 0)), (0, 0)))
    x = x[:, max(-pady0, 0) : H - max(-pady1, 0), max(-padx0, 0) : W - max(-padx1, 0)]

    #setup filter
    f = f * (gain ** (f.ndim / 2))
    if not flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)
    
    #convolve filter
    x = jnp.repeat(jnp.expand_dims(f, axis=(-2, -1)), repeats=C, axis=-1)
    x = jax.lax.conv_general_dilated(x, f,
                                     window_strides=strides or (1,) * (x.ndim-2),
                                     padding='valid',
                                     dimension_numbers=nn.linear._conv_dimension_numbers(x.shape),
                                     feature_group_count=C)
    x = x[:, ::down, ::down]
    return x


class LinearLayer(nn.Module):
    """Linear Layer.
    Attributes:
        in_features (int): Input dim.
        out_features (int): Output dim.
        use_bias (bool): If True, use bias.
        bias_init (int): Bias init.
        lr_multiplier (float): lr multiplier.
        activation (str): Activation function.
        param_dict (h5py.Group): Param dict with pretrained params.
        layer_name (str): layer name.
        dtype (str): data type.
        rng (jax.random.PRNGKey): Random seed.
    """
    in_features: int
    out_features: int
    use_bias: bool = True
    bias_init: int = 0
    lr_multiplier: float = 1
    activation: str = 'linear'
    param_dict: Optional[h5py.Group] = None
    layer_name: str = 'linear'
    dtype: str = 'float32'
    rng: jax.random.PRNGKey = jax.random.PRNGKey(42)

    @nn.compact
    def __call__(self, x):
        """
        Run Linear Layer.
        Args:
            x (Tensor): input tensor of shape [N, in_features].
        Returns:
            (tensor): Output tensor of shape [N, out_features].
        """
        w_shape = [self.in_features, self.out_features]
        params = get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng, self.dtype)
        if self.use_bias:
            w, b = params
        else:
            w = params
        w = self.param(name='weight', init_fn=lambda *_: w)
        w = equalize_lr_weights(w, self.lr_multiplier)
        x = jnp.matmul(x, w)
        if self.use_bias:
            b = self.param(name='bias', init_fn=lambda *_: b)
            b = equalize_lr_bias(b, self.lr_multiplier)
            x += b
            x += self.bias_init

        x = apply_activation(x, self.activation)
        return x


def linear_layer(x, w, b, activation='Linear'):
    x = jnp.matmul(x, w)
    if b is not None:
        x += b
    x = apply_activation(x, activation)
    return x    


def conv_downsample_2d(x, w, k=None, factor=2, gain=1, padding=0):
    """Fused downsample convolution.
    Padding is performed only at beginning, not between operations.
    The fused op is more efficient than performing the same calculation
    using tf ops. It supports gradients of arbitrary order.
    
    Args:
        x (tensor): Input of shape [N, H, W, C].
        w (tensor): Weight of shape [filterH, filterW, inChannels, outChannels].
                    grouped conv can be performed by inChannels = x.shape[0] // numGroups.
        k (tensor): FIR filter of shape [firH, firW] or [firN].
        factor (int): Downsample factor.
        gain (float): Scaling factor for signal magnitude.
        padding (int): Number of pixels to pad on each side.
    
    Returns:
        (tensor): Output of shape [N, H // factor, W // factor, C].
    """
    assert isinstance(factor, int) and factor >= 1, 'factor must be an integer >= 1'
    assert isinstance(padding, int), 'padding must be an integer'

    # check weight shape.
    ch, cw, _intC, _outC = w.shape
    assert ch == cw

    # setup filter kernel.
    k = setup_filter(k, gain)
    assert k.shape[0] == k.shape[1]

    # execute.
    pad0 = (k.shape[0] - factor + cw) // 2 + padding * factor
    pad1 = (k.shape[0] - factor + ch - 1) // 2 + padding * factor
    x = upfirdn2d(x, w, k, factor, (pad0, pad0, pad1, pad1))

    x = jax.lax.conv_general_dilated(x, w,
                                     window_strides=(factor, factor),
                                     padding='VALID',
                                     dimension_numbers=nn.Linear._conv_dimension_numbers(x.shape))

    return x


