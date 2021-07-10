import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List
import h5py
from . import ops
from .. import utils

class MappingNetwork(nn.Module):
  
    # Dimensionality
    z_dim: int=512
    c_dim: int=0
    w_dim: int=512
    embed_features: int=None
    layer_features: int=512

    # Layers
    num_ws: int=18
    num_layers: int=8
    
    # Pretrained
    pretrained: str=None
    param_dict: h5py.Group=None
    ckpt_dir: str=None

    # Internal details
    activation: str='leaky_relu'
    lr_multiplier: float=0.01
    w_avg_beta: float=0.995
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)
    
    def setup(self):
        self.embed_features_ = self.embed_features
        self.c_dim_ = self.c_dim
        self.layer_features_ = self.layer_features
        self.num_layers_ = self.num_layers
        self.param_dict_ = self.param_dict

        if self.pretrained is not None and self.param_dict is None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['mapping_network']
            self.c_dim_ = C_DIM[self.pretrained]
            self.num_layers_ = NUM_MAPPING_LAYERS[self.pretrained]

        if self.embed_features_ is None:
            self.embed_features_ = self.w_dim
        if self.c_dim_ == 0:
            self.embed_features_ = 0
        if self.layer_features_ is None:
            self.layer_features_ = self.w_dim

        if self.param_dict_ is not None and 'w_avg' in self.param_dict_:
            self.w_avg =  self.variable('moving_stats', 'w_avg', lambda *_ : jnp.array(self.param_dict_['w_avg']), [self.w_dim])
        else:
            self.w_avg = self.variable('moving_stats', 'w_avg', jnp.zeros, [self.w_dim])
