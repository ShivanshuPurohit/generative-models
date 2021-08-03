from _typeshed import Self
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List
import h5py
from . import ops, utils

URLS = {'afhqwild': 'https://www.dropbox.com/s/p1slbtmzhcnw9q8/stylegan2_generator_afhqwild.h5?dl=1',
        'brecahad': 'https://www.dropbox.com/s/28uykhj0ku6hwg2/stylegan2_generator_brecahad.h5?dl=1',
        'car': 'https://www.dropbox.com/s/67o834b6xfg9x1q/stylegan2_generator_car.h5?dl=1'}


class MappingNetwork(nn.Module):
    z_dim: int = 512
    w_dim: int = 512
    embed_features: int = None
    layer_features: int = 512

    num_ws: int = 18
    num_layers: int = 8

    pretrained: str = None
    param_dict: h5py.Group = None
    ckpt_dir: str = None

    activation: str = 'leaky_relu'
    lr_multiplier: float = 0.01
    w_avg_beta: float = 0.995
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    def setup(self):
        self.embed_features_ = self.embed_features
        self.c_dim_ = self.c_dim
        self.layer_features_ = self.layer_features
        self.num_layers_ = self.num_layers
        self.param_dict_ = self.param_dict
        if self.pretrained is not None and self.param_dict is None:
            assert self.pretrained in URLS.keys(), f'Pretrained model unavailable: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir,  URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['mapping_network']
        if self.embed_features_ is None:
            self.embed_features_ = self.w_dim
        if self.layer_features_ is None:
            self.layer_features_ = self.w_dim
        if self.param_dict_ is not None and 'w_avg' in self.param_dict_:
            self.w_avg = self.variable('moving_stats', 'w_avg', lambda*_: jnp.array(self.param_dict_['w_avg']), [self.w_dim])
    
    @nn.compact
    def __call__(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True):
        """
        Run Mapping network.
        Args:
            z: Input Noise, shape [N, z_dim]
            c: Input Labels, shape [N, c_dim]
            truncation_psi: Truncation (variation vs quality tradeoff) 1 means off
            skip_w_avg_update: if true, updates ema of W
            train: to train or not
        Returns:
            tensor: intermediate tensor W
        """
        x = None
        if self.z_dim > 0:
            x = ops.normalize_2nd_moment(z)
        for i in range(self.num_layers_):
            x = ops.LinearLayer(in_features=x.shape[1],
                                out_features=self.layer_features_,
                                use_bias=True,
                                lr_multiplier=self.lr_multiplier,
                                activation=self.activation,
                                param_dict=self.param_dict_,
                                layer_name=f'fc{i}',
                                dtype=self.dtype,
                                rng=self.rng)(x)
        if self.w_avg_beta is not None and train and not skip_w_avg_update:
            self.w_avg.value = self.w_avg_beta * self.w_avg.value + (1 - self.w_avg_beta) * jnp.mean(x, axis=0)
        if self.num_ws is not None:
            x = jnp.repeat(jnp.expand_dims(x, axis=-2), repeats=self.num_ws, axis=-2)
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = truncation_psi * x + (1 - truncation_psi) * self.w_avg.value
            else:
                x[:, :truncation_cutoff] = truncation_psi * x[:, :truncation_cutoff] + (1 - truncation_psi) * self.w_avg.value
        return x


class SynthesisLayer(nn.Module):
    fmaps: int
    kernel: int
    layer_idx: int
    lr_multiplier: float = 1
    up: bool = False
    activation: str = 'leaky_relu'
    use_noise: bool = True
    randomize_noise: bool = True
    resample_kernel: Tuple = (1, 3, 3, 1)
    fused_modconv: bool = False
    param_dict: h5py.Group = None
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    @nn.compact
    def __call__(self, x, dlatents):
        """
        Run Synthesis Layer.
        Args:
            x: Input tensor of shape [N, H, W, C]
            dlatents: Intermediate latents of shape [N, num_ws, w_dim]
        Returns:
            tensor: output tensor of shape [N, H', W', fmaps]
        """
        s = ops.LinearLayer(in_features=dlatents[:, self.layer_idx].shape[1],
                            out_features=x.shape[3],
                            use_bias=True,
                            bias_init=1,
                            lr_multiplier=self.lr_multiplier,
                            param_dict=self.param_dict,
                            layer_name='affine',
                            dtype=self.dtype,
                            rng=self.rng)(dlatents[:, self.layer_idx])
        if self.param_dict is None:
            noise_strength = jnp.zeros(())
        else:
            noise_strength = jnp.array(self.param_dict['noise_strength'])
        noise_strength = self.param(name='noise_strength', init_fn=lambda *_: noise_strength)

        # weight and bias for convolution operation:
        w_shape = [self.kernel, self.kernel, x.shape[3], self.fmaps]
        w, b = ops.get_weight(w_shape, self.lr_multiplier, True, self.param_dict, self.rng, name='conv', dtype=self.dtype)
        w = self.param(name='weight', init_fn=lambda *_: w)
        b = self.param(name='bias', init_fn=lambda *_: b)
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)

        x = ops.modulated_conv2d_layer(x=x,
                                       w=w,
                                       s=s,
                                       fmaps=self.fmaps,
                                       kernel=self.kernel,
                                       up=self.up,
                                       resample_kernel=self.resample_kernel,
                                       fused_modconv=self.fused_modconv)
        if self.use_noise:
            if not self.randomize_noise and self.param_dict is not None:
                noise = jnp.array(self.param_dict['noise_const'])
            else:
                noise = random.normal(self.rng, shape=(x.shape[0], x.shape[1], x.shape[2], 1), dtype=self.dtype)
            x += noise_strength.astype(self.dtype) * noise
        x += b
        x = ops.apply_activation(x, self.activation)
        return x


class ToRGBLayer(nn.Module):
    """
    To RGB Layer
    fmaps (int): Number of output channels to modconv
    layer_idx (int): Layer index to access latent code for a specific layer
    kernel (int): Kernel size of modconv
    fused_modconv (bool): If True, use fused modconv
    param_dict (h5py.Group): h5py Group containing weight parameters
    dtype (str): Data type of weights
    rng (jax.random.PRNGKey): PRNG key to initialize weights
    """
    fmaps: int
    layer_idx: int
    kernel: int = 1
    lr_multiplier: float = 1
    activation: str = 'leaky_relu'
    use_noise: bool = True
    randomize_noise: bool = True
    resample_kernel: Tuple = (1, 3, 3, 1)
    fused_modconv: bool = False
    param_dict: h5py.Group = None
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    @nn.compact
    def __call__(self, x, y, dlatents):
        """
        Run ToRGB Layer.
        Args:
            x: Input tensor of shape [N, H, W, C]
            y: Image of shape [N, H', W', C]
            dlatents: Intermediate latents of shape [N, num_ws, w_dim]
        Returns:
            tensor: output tensor of shape [N, H', W', C]
        """
        s = ops.LinearLayer(in_features=dlatents[:, self.layer_idx].shape[1],
                            out_features=x.shape[3],
                            use_bias=True,
                            bias_init=1,
                            lr_multiplier=self.lr_multiplier,
                            param_dict=self.param_dict,
                            layer_name='affine',
                            dtype=self.dtype,
                            rng=self.rng)(dlatents[:, self.layer_idx])

        # weight and bias for convolution operation:
        w_shape = [self.kernel, self.kernel, x.shape[3], self.fmaps]
        w, b = ops.get_weight(w_shape, self.lr_multiplier, True, self.param_dict, self.rng, name='conv', dtype=self.dtype)
        w = self.param(name='weight', init_fn=lambda *_: w)
        b = self.param(name='bias', init_fn=lambda *_: b)
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)
        x = ops.modulated_conv2d_layer(x, w, s, fmaps=self.fmaps, kernel=self.kernel, demodulate=False, fused_modconv=self.fused_modconv)
        x += b
        x = ops.apply_activation(x, activation='linear')
        if y is not None:
            x += y
        return x


class SynthesisBlock(nn.Module):
    """
    Synthesis Block
    Attributes:
    fmaps (int): number of output channels of the modconv
    res (int): Resolution (log2) of the current block
    num_layers (int): Number of layers in the current block
    num_channels (int): Number of output color channels
    activation (str): Activation function: 'relu', 'lrelu', etc
    lr_multiplier (float): LR Multiplier
    param_dict (h5py.Group): Param dict with pretrained model parameters
    dtype (str): Data type of weights
    """
    fmaps: int
    res: int
    num_layers: int
    num_channels: int
    activation: str = 'leaky_relu'
    use_noise: bool = True
    randomize_noise: bool = True
    fused_modconv: bool = False
    resample_kernel: Tuple = (1, 3, 3, 1)
    lr_multiplier: float = 1
    param_dict: h5py.Group = None
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    @nn.compact
    def __call__(self, x, y, dlatents_in):
        """
        Run Synthesis Block.
        Args:
            x: Input tensor of shape [N, H, W, C]
            dlatents: Intermediate latents of shape [N, num_ws, w_dim]
            y: Image of shape [N, H', W', C]
        Returns:
            tensor: output tensor of shape [N, H', W', C]
        """
        for i in range(self.num_layers):
            x = SynthesisLayer(fmaps=self.fmaps,
                               kernel=3,
                               layer_idx=self.res * 2 - (5 - i) if self.res > 2 else 0,
                               up=i == 0 and self.res != 2,
                               use_noise=self.use_noise,
                               randomize_noise=self.randomize_noise,
                               resample_kernel=self.resample_kernel,
                               fused_modconv=self.fused_modconv,
                               activation=self.activation,
                               lr_multiplier=self.lr_multiplier,
                               param_dict=self.param_dict[f'layer{i}'] if self.param_dict is not None else None,
                               dtype=self.dtype,
                               rng=self.rng)(x, dlatents_in)
        
        if self.num_layers == 2:
            k = ops.setup_filter(self.resample_kernel, gain=2**2)
            padx0 = (k.shape[0] + 1) // 2
            padx1 = (k.shape[0] - 2) // 2
            pady0 = (k.shape[1] + 1) // 2
            pady1 = (k.shape[1] - 2) // 2
            y = ops.upfirdn2d(y, f=k, padding=(padx0, padx1, pady0, pady1))
    
        y = ToRGBLayer(fmaps=self.num_channels,
                       layer_idx=self.res * 2 - 3,
                       lr_multiplier=self.lr_multiplier,
                       param_dict=self.param_dict['torgb'] if self.param_dict is not None else None,
                       dtype=self.dtype,
                       rng=self.rng)(x, y, dlatents_in)
        return x, y


class SynthesisNetwork(nn.Module):
    """
    Synthesis Network
    Attributes:
        resolution (int): Output res
        num_channels (int): number of output color channels
        w_dim (int): input latent (z) dimensionality
        fmap_base (int): overall multiplier for the number of feature maps
        fmap_decay (int): log2 feature map reduction per block
        fmap_max (int): maximum number of feature maps
        fmap_min (int): minimum number of feature maps
        use_noise (bool): Use noise in the network
        fmap_const (int): Number if feature maps in constant input layer
        randomize_noise (bool): Randomize
        resample_kernel (Tuple): Kernel that is used for FIR Filter
        activation (str): activation function
        param_dict (h5py.Group): Param dict with pretrained params
        dtype (str): data type
        rng (jax.random.PRNGKey): random seed
    """
    resolution: int = 1024
    num_channels: int = 3
    w_dim: int = 512
    fmap_base: int = 16384
    fmap_decay: int = 1
    fmap_min: int = 1
    fmap_max: int = 512
    fmap_const: int = None
    param_dict: h5py.Group = None
    activation: str = 'leaky_relu'
    use_noise: bool = True
    randomize_noise: bool = True
    resample_kernel: Tuple = (1, 3, 3, 1)
    fused_modconv: bool = False
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    def setup(self):
        self.resolution_ = self.resolution
        self.param_dict_ = self.param_dict
    
    @nn.compact
    def __call__(self, dlatents_in):
        """
        Run Synthesis network
        Args:
            dlatents_in: Intermediate latents (W) of shape [N, num_ws, w_dim]
        Returns:
            (tensor): image of shape [N, H, W, num_channels]
        """
        resolution_log2 = int(np.log2(self.resolution_))
        assert self.resolution_ == 2 ** resolution_log2 and self.resolution_ >= 4
        def nf(stage): return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)
        num_layers = resolution_log2 * 2 - 2
        fmaps = self.fmap_const if self.fmap_const is not None else nf(1)
        if self.param_dict_ is None:
            const = random.normal(self.rng, (1, 4, 4, fmaps))
        else:
            const = jnp.array(self.param_dict_['const'])
        x = self.param(name='const', init_fn=lambda *_: const)
        x = jnp.repeat(x, repeats=dlatents_in.shape[0], axis=0)
        y = None
        for res in range(2, resolution_log2 + 1):
            x = SynthesisLayer(fmaps=nf(res-1),
                               res=res,
                               num_layers=1 if res==2 else 2,
                               use_noise=self.use_noise,
                               randomize_noise=self.randomize_noise,
                               resample_kernel=self.resample_kernel,
                               fused_modconv=self.fused_modconv,
                               activation=self.activation,
                               param_dict=self.param_dict[f'block{2 ** res}x{2 ** res}'] if self.param_dict is not None else None,
                               dtype=self.dtype,
                               rng=self.rng)(x, y, dlatents_in)
        return y


class Generator(nn.Module):
    """
    Generator
    Attributes:
        resolution (int): resolution
        num_channels (int): number of output color channels
        z_dim (int): Input latent Z dimensionality
        w_dim (int): Intermediate latent W dimensionality
        mapping_layer_features (int): Number of intermediate features in mapping layers
        mapping_embed_features (int): Label embedding dimensionality
        num_ws (int): Number of intermediate latents to output
        num_mapping_layers (int): Number of mapping layers
        fmap_base (int): Overall multiplier for the number of feature maps
        fmap_decay (int): Log2 feature map reduction when doubling the resolution
        fmap_min (int): Min number of feature maps in a layer
        fmap_max (int): Max number of feature maps in a layer
        fmap_const (int): Number of feature maps in const input layer
        use_noise (bool): if True, add spatial-specific noise
        randomize_noise (bool): if True, use random noise
        activation (str): activation function
        w_avg_beta (float): decay for tracking moving avg of W during training
    """
    resolution: int = 1024
    num_channels: int = 3
    z_dim: int = 512
    w_dim: int = 512
    mapping_layer_features: int = 512
    mapping_embed_features: int = None
    num_ws: int = 18
    num_mapping_layers: int = 8
    use_noise: bool = True
    randomize_noise: bool = True
    w_avg_beta: float = 0.995
    activation: str = 'leaky_relu'
    resample_kernel: Tuple = (1, 3, 3, 1)
    fused_modconv: bool = False
    dtype: str = 'float32'
    rng: Any = random.PRNGKey(42)

    @nn.compact
    def __call__(self, z, truncation_psi=1, truncation_cutoff=None, skip__avg_update=False, train=True):
        """
        Run Generator
        Args:
            z: input noise, shape [N, z_dim]
            truncation_psi (float): controls truncation, 1 means off
            truncation_cutoff (int): controls truncation, None means off
            skip_w_avg_update (bool): if True, updates the ema of W
            train (bool): training flag
        Returns:
            (tensor): image of shape [N, H, W, num_channels] 
        """
        dlatents_in = MappingNetwork(z_dim=self.z_dim,
                                     w_dim=self.w_dim,
                                     num_ws=self.num_ws,
                                     num_layers=self.num_mapping_layers,
                                     embed_features=self.mapping_embed_features,
                                     layer_features=self.mapping_layer_features,
                                     activation=self.activation,
                                     lr_multiplier=self.lr_multiplier,
                                     w_avg_beta=self.w_avg_beta,
                                     param_dict=self.param_dict['mapping_network'] if self.param_dict is not None else None,
                                     dtype=self.dtype,
                                     rng=self.rng)(z, truncation_psi, truncation_cutoff, skip__avg_update, train)
        
        x = SynthesisLayer(resolution=self.resolution,
                           num_channels=self.num_channels,
                           w_dim=self.w_dim,
                           fmap_base=self.fmap_base,
                           fmap_decay=self.fmap_decay,
                           fmap_min=self.fmap_min,
                           fmap_max=self.fmap_max,
                           fmap_const=self.fmap_const,
                           param_dict=self.param_dict['synthesis_network'] if self.param_dict is not None else None,
                           activation=self.activation,
                           resample_kernel=self.resample_kernel,
                           randomize_noise=self.randomize_noise,
                           fused_modconv=self.fused_modconv,
                           dtype=self.dtype,
                           rng=self.rng)(dlatents_in)
        
        return x
