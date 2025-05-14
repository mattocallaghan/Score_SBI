import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import jax
from . import utils, layers
get_act = layers.get_act
default_initializer = layers.default_init
from models.layerspp import GaussianFourierProjection
default_initializer = layers.default_init

@utils.register_model(name='simple_score')
class SimpleScoreNet(nn.Module):
    """A simple fully connected score network with continuous time embedding."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True):
        """
        while this says time condition herre, in reality it is the noise level as when called to the
        score function we pass through to get sigmas first
        """

        config = self.config
        dropout = config.model.dropout

        act = get_act(config)
        sigmas = utils.get_sigmas(config)

        nf = config.model.nf  #embedding size
        conditional = config.model.conditional  # noise-conditional

        # x: shape (batch, n_features)
        # time_cond: continuous time values, shape (batch,) or (batch, 1)
        if time_cond.ndim == 1:
            time_cond = time_cond[:, None]
        # Configuration parameters.
        hidden_size = config.model.hidden_size  
        num_layers = config.model.num_layers   
        fourier_scale = config.model.fourier_scale
        time_dense_size = config.model.time_dense_size

        # this keeps it bounded

        # Continuous time embedding using Gaussian Fourier features.

        # Optionally, further process the embedding with a Dense layer.
        if(conditional):
            if(config.model.embedding_type == 'fourier'):
                used_sigmas=time_cond
                time_emb = GaussianFourierProjection(
                            embedding_size=nf, scale=fourier_scale)(jnp.log(time_cond))
            elif config.model.embedding_type == 'positional':
                # Sinusoidal positional embeddings.
                used_sigmas = sigmas[time_cond.astype(jnp.int32)]
                time_emb = layers.get_timestep_embedding(used_sigmas, nf)
            else:
                raise ValueError(f'Unknown embedding type {config.model.embedding_type}.')

        else:
            time_emb=None
            used_sigmas=time_cond
            
        if(conditional):
                time_emb = nn.Dense(4*nf, name=f'time_embedding_dense_init',use_bias=False)(time_emb)
                time_emb = nn.Dense(time_dense_size, name=f'time_embedding_dense_1',use_bias=False)(act(time_emb))

        h=jnp.concatenate([x, time_emb], axis=-1) if conditional else x

        for i in range(num_layers - 1):
            h = nn.Dense(hidden_size, name=f'hidden_dense_{i}',use_bias=True)(h)
            h = nn.Dropout(dropout)(h, deterministic=not train)
            h = act(h)



        out = nn.Dense(x.shape[-1],name='output_dense',use_bias=True)(h)
        if config.model.scale_by_sigma:
            out=out/used_sigmas.reshape(-1,1)
        return out


