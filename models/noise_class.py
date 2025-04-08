import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import jax
import models.nutils as nutils
from models.layerspp import GaussianFourierProjection

@nutils.register_noise_model(name='simple_noise')
class register_noise_model(nn.Module):
    config: ml_collections.ConfigDict
    @nn.compact
    def __call__(self, x,train=True):
        config = self.config
        dropout = config.noise_model.dropout
        N = config.model.num_scales
        hidden_size = self.config.noise_model.hidden_size

        x = nn.sigmoid(nn.Dense(10)(x))
        x = nn.Dropout(dropout)(x, deterministic=not train)

        x=(nn.Dense(N)(x))        
        return x

