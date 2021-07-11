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
        