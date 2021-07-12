import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List, Callable
import h5py
from . import ops, utils


class FromRGBLayer(nn.Module):
    """
    From RGB Layer
    Attributes:
        fmaps (int): number of output channels of convolution
        kernel (int): Kernel size of the convolution
        lr_multiplier (float): lr multiplier
        activation (str): activation function
        param_dict (h5py.Group): Param dict with pretrained params
    """
    fmaps: int
    kernel: int = 1
    lr_multiplier: float = 1.0
    activation: str = 'leaky_relu'
    param_dict: h5py.Group = None
    
    @nn.compact
    def __call__(self, x, y):
        """
        Run from RGB Layer.
        Args:
            x: Input image of shape [N, H, W, num_channels]
            y: Input tensor of shape [N, H, W, num_channels]
        Return:
            Output tensor of shape [N, H, W, num_channels]
        """
        w_shape = [self.kernel, self.kernel, x.shape[-1], self.fmaps]
        w, b = ops.get_weight(w_shape, self.lr_multiplier, True, self.param_dict, 'fromrgb')
        w = self.param(name='weight', init_fn=lambda *_: w)
        b = self.param(name='bias', init_fn=lambda *_: b)
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)
        x = ops.conv2d(x, w)
        x += b
        x = ops.apply_activation(x, self.activation)
        if y is not None:
            x += y
        return x


class DiscriminatorLayer(nn.Module):
    """
    Discriminator Layer
    Attributes:
        fmaps (int): number of output channels of convolution
        kernel (int): Kernel size of the convolution
        use_bias (bool): Use bias in convolution
        down (bool): Use downsampling for spatial resolution if True
        resample_kernel (Tuple[int, int]): Kernel size for FIR filter
        layer_name (str): Name of the layer
        lr_multiplier (float): lr multiplier
        activation (str): activation function
        param_dict (h5py.Group): Param dict with pretrained params
        dtype (str): Data type
        rng (jax.random.PRNGKey): Random seed
    """
    fmaps: int
    kernel: int = 3
    use_bias: bool = True
    down: bool = False
    resample_kernel: Tuple = None
    layer_name: str = None
    lr_multiplier: float = 1.0
    activation: str = 'leaky_relu'
    param_dict: h5py.Group = None
    dtype: str = 'float32'
    rng: Any= jax.random.PRNGKey(42)
    
    @nn.compact
    def __call__(self, x):
        """
        Run Discriminator Layer.
        Args:
            x: Input image of shape [N, H, W, num_channels]
        Return:
            Output tensor of shape [N, H, W, num_channels]
        """
        w_shape = [self.kernel, self.kernel, x.shape[-1], self.fmaps]
        if self.use_bias:
            w, b = ops.get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng)
        else:
            w = ops.get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng)
        w = self.param(name='weight', init_fn=lambda *_: w)
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        if self.use_bias:
            b = self.param(name='bias', init_fn=lambda *_: b)    
            b = ops.equalize_lr_bias(b, self.lr_multiplier)
        x = ops.conv2d(x, w, down=self.down, resample_kernel=self.resample_kernel)
        if self.use_bias:
            x += b
        x = ops.apply_activation(x, self.activation)
        return x


class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block
    Attributes:
        fmaps (int: Number of output channels of the conv
        kernel (int): Kernel size of the conv
        resample_kernel (Tuple): Kernel used for FIR Filter
        activation (str): activation function
        param_dict (h5py.Group): param dict with pretrained params
        lr_multiplier (float): lr multiplier
        nf (Callable): callable that returns the number of fmaps for a layer
        architecture (str): 'resnet' or 'orig'
        dtype (str): data type
        rng (jax.random.PRNGKey): Random seed
    """
    fmaps: int
    kernel: int = 3
    resample_kernel: Tuple = (1, 3, 3, 1)
    activation: str = 'leaky_relu'
    param_dict: Any = None
    lr_multiplier: float = 1.0
    nf: Callable = None
    architecture: str = 'resnet'
    dtype: str = 'float32'
    rng: Any= jax.random.PRNGKey(42)
    
    @nn.compact
    def __call__(self, x):
        """
        Run Discriminator Block.
        Args:
            x: Input image of shape [N, H, W, num_channels]
        Return:
            Output tensor of shape [N, H, W, fmaps]
        """
        residual = x
        for i in range(2):
            x = DiscriminatorLayer(fmaps=self.nf(self.res-(i+1)),
                                   kernel=self.kernel,
                                   down=i==1,
                                   resample_kernel=self.resample_kernel if i==1 else None,
                                   activation=self.activation,
                                   layer_name=f'conv{i}',
                                   lr_multiplier=self.lr_multiplier,
                                   param_dict=self.param_dict,
                                   dtype=self.dtype,
                                   rng=self.rng)(x)
        if self.architecture == 'resnet':
            x = DiscriminatorLayer(fmaps=self.nf(self.res-(i+1)),
                                   kernel=self.kernel,
                                   down=i==1,
                                   resample_kernel=self.resample_kernel if i==1 else None,
                                   activation='Linear',
                                   layer_name='skip',
                                   lr_multiplier=self.lr_multiplier,
                                   param_dict=self.param_dict,
                                   dtype=self.dtype,
                                   rng=self.rng)(residual)
            x = (x + residual) * (1 / np.sqrt(2))
        return x


class Discriminator(nn.Module):
    """
    Discriminator
    Attributes:
        resolution (int): Input resolution
        num_channels (int): Number of input color channels
        fmap_base(int): Base number of output channels
        fmap_decay (int): Decay of number of output channels
        fmap_max (int): Maximum number of output channels
        fmap_min (int): Minimum number of output channels
        mapping_layers (int): Number of mapping layers for conditioning labels
        mapping_fmaps (int): Number of activations in mapping layers
        architecture (str): 'resnet' or 'orig'
        activation (str): activation function
        mbstd_group_size (int): Size of minibatch standard deviation layer
        mbstd_num_features (int): Number of features in minibatch standard deviation layer
        resample_kernel (Tuple): Kernel used for FIR Filter
        dtype (str): Data type
        rng (jax.random.PRNGKey): Random seed
    """
    resolution: int = 1024
    num_channels: int = 3
    fmap_base: int = 16384
    fmap_decay: int = 1
    fmap_max: int = 512
    fmap_min: int = 1
    mapping_layers: int = 0
    mapping_fmaps: int = None
    mapping_lr_multiplier: float = 0.1
    architecture: str = 'resnet'
    activation = 'leaky_relu'
    mbstd_group_size: int = None
    mbst_num_features: int = 1
    resample_kernel: Tuple = (1, 3, 3, 1)
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    def setup(self):
        self.resolution_ = self.resolution
        self.architecture_ = self.architecture
        self.mbstd_group_size_ = self.mbstd_group_size
        self.param_dict = None

    @nn.compact
    def __call__(self, x):
        """
        Run discriminator
        Args:
            x: Input image of shape [N, H, W, num_channels]
        Returns:
            output tensor of shape [N, 1]
        """
        resolution_log2 = int(np.log2(self.resolution_))
        assert self.resolution_ == 2 ** resolution_log2 and self.resolution_ >= 4
        def nf(stage): return np.clip(int(self.fmap_base/(2.0**(stage*self.fmap_decay))), self.fmap_min, self.fmap_max)
        if sef.mapping_fmaps is None:
            mapping_fmaps = nf(0)
        else:
            mapping_fmaps = self.mapping_fmaps

        #layers for >= 8*8 resolution
        y = None
        for res in range(resolution_log2, 2, -1):
            res_str = f'block_{2**res}x{2**res}'
            if res == resolution_log2:
                x = FromRGBLayer(fmaps=nf(res-1),
                                 kernel=1,
                                 activation=self.activation,
                                 param_dict=self.param_dict[res_str] if self.param_dict is not None else None,
                                 dtype=self.dtype,
                                 rng=self.rng)(x, y)
            
            x = DiscriminatorBlock(res=res,
                                   kernel=3,
                                   resample_kernel=self.resample_kernel,
                                   activation=self.activation,
                                   parsam_dict=self.param_dict[res_str] if self.param_dict is not None else None,
                                   architecture=self.architecture_,
                                   nf=nf,
                                   dtype=self.dtype,
                                   rng=self.rng)(x)

        x = x.astype(self.dtype)
        if self.mbst_num_features > 0:
            x = ops.minibatch_stddev_layer(x, self.mbstd_group_size_, self.mbstd_num_features)
        x = DiscriminatorLayer(fmaps=mapping_fmaps,
                               kernel=3,
                               activation=self.activation,
                               use_bias=True,
                               layer_name='conv0',
                               param_dict=self.param_dict['block_4x4'] if self.param_dict is not None else None,
                               dtype=self.dtype,
                               rng=self.rng)(x)
        #switch to [N, C, H, W] so that pretrained weights still work
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, newshape=(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        x = ops.LinearLayer(in_features=x.shape[1],
                            out_features=nf(0),
                            param_dict=self.param_dict['block_4x4'] if self.param_dict is not None else None,
                            layer_name='fc0',
                            dtype=self.dtype,
                            rng=self.rng)(x)
        # output layer
        x = ops.LinearLayer(in_features=x.shape[1],
                            out_features=mapping_fmaps,
                            param_dict=self.param_dict,
                            layer_name='output',
                            dtype=self.dtype,
                            rng=self.rng)(x)
        return x