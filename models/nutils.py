# noise utils
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import numpy as np
from flax.training import checkpoints
from utils import batch_mul
import optax


@flax.struct.dataclass
class NoiseState:
  step: int
  opt_state: Any
  lr: float
  model_state: Any
  ema_rate: float
  params_ema: Any
  rng: Any
  params: Any







_NOISEMODELS = {}


def register_noise_model(cls=None, *, name=None):
  """A decorator for registering noise model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _NOISEMODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _NOISEMODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_noise_model(name):
  return _NOISEMODELS[name]




def init_noise_model(rng, config):
  """ Initialize a `flax.linen.Module` noise model. """
  #labels here arent a thing
  model_name = config.noise_model.name
  model_def = functools.partial(get_noise_model(model_name), config=config)
  input_shape = (jax.local_device_count(), config.data.vector_dim)
  fake_input = jnp.zeros(input_shape)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = (model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input))
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  # i think mine is not a forzen dictionarty
  # i cast it directly to one!
  # mine is not a frozen dictionary for now so i will leave it as not


  init_model_state, initial_params = flax.core.pop(variables,"params")
  return model, init_model_state, initial_params





def get_noise_model_fn(model, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def noise_model_fn(x, rng):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = {'params': params, **states}
    if not train:
      return model.apply(variables, x, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, x, train=True, mutable=list(states.keys()), rngs=rngs)

  return noise_model_fn






@jax.jit
def sample_joint(x, sigma_max, sigma_min,rng,ts):
    rng2,rng3 = jax.random.split(rng,2)
    #need to update for other models
    ss = sigma_min * (sigma_max / sigma_min) ** ts
    y = jax.random.randint(rng2, (1,), 0, len(ts))
    
    x = x + ss[y] * jax.random.normal(rng3, x.shape)

    return x,y #should be always positive