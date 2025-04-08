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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import scipy.integrate as integrate
import jax
import jax.numpy as jnp
import jax.random as random
import equinox
import abc
import flax
import diffrax
import equinox as eqx
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
import sde_lib
from utils import batch_mul, batch_add

from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, model, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  model=model,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 model=model,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps)

  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the predictor.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the corrector.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    dt = -1. / self.rsde.N
    z = random.normal(rng, x.shape)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    f, G = self.rsde.discretize(x, t)
    z = random.normal(rng, x.shape)
    x_mean = x - f
    x = x_mean + batch_mul(G, z)
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
    std = jnp.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, noise)
    return x, x_mean

  def vpsde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    beta = sde.discrete_betas[timestep]
    score = self.score_fn(x, t)
    x_mean = batch_mul((x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(jnp.sqrt(beta), noise)
    return x, x_mean

  def update_fn(self, rng, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(rng, x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(rng, x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, rng, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      grad_norm = jnp.linalg.norm(
        grad.reshape((grad.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean





@register_corrector(name='langevin_time_exact')
class LangevinCorrectorTimeExact(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

#assumes a sigma prior of 1/sigma
  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    rsde = sde.reverse(score_fn, probability_flow=True)
    drift_fn = lambda X, T: (rsde.sde)(X,T)[0]
    sigma_t = lambda T: sde.marginal_prob(jnp.zeros_like(x), jnp.ones(x.shape[0]) * T)[1]
    grad_sigma_t = (jax.grad(lambda T: sde.marginal_prob(1.0, T)[1]))

    

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean,t_in,t_in_mean = val
      grad = score_fn(x, t_in)

      
      jacobian = jax.vmap(jax.jacfwd(drift_fn,argnums=0))(x, t_in[:,None])  # Differentiate w.r.t. x only

      

      if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        #quite an unstable calculation for vesde
        dsigma_dt=jax.vmap(grad_sigma_t)(t_in)[:,None]
        d2sigma_dt2 = jax.vmap(jax.jacfwd(lambda T: grad_sigma_t(T)))(t_in)[:,None]
        dt = jnp.diagonal(jacobian, axis1=-2, axis2=-1).sum(-1) - (1.0 / sigma_t(t_in))[:,None] * dsigma_dt +   d2sigma_dt2/jnp.abs(dsigma_dt)
      else:
        dsigma_dt=jax.vmap(grad_sigma_t)(t_in)[:,None]
        dt =-jnp.diagonal(jacobian, axis1=-2, axis2=-1).sum(-1)#-1e3*dsigma_dt
      grad_xt=jnp.concatenate([grad,dt],axis=-1)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, grad_xt.shape)
      grad_norm = jnp.linalg.norm(
        grad_xt.reshape((grad_xt.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      
      xt=jnp.concatenate([x,t_in[:,None]],axis=-1)
      xt_mean = xt + batch_mul(step_size, grad_xt)
      xt = xt_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, xt[:,:-1], xt_mean[:,:-1],xt[:,-1],xt_mean[:,-1]

    _, x, x_mean,t,t_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x,t,t))
    return x, x_mean,t,t_mean




@register_corrector(name='langevin_time_approx')
class LangevinCorrectorTimeApprox(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

#assumes a sigma prior of 1/sigma
  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    rsde = sde.reverse(score_fn, probability_flow=True)
    drift_fn = lambda X, T: (rsde.sde)(X,T)[0]
    sigma_t = lambda T: sde.marginal_prob(jnp.zeros_like(x), jnp.ones(x.shape[0]) * T)[1]
    grad_sigma_t = (jax.grad(lambda T: sde.marginal_prob(1.0, T)[1]))



    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean,t_in,t_in_mean = val
      grad = score_fn(x, t_in)

      rng, step_rng = random.split(rng)
      eps = jax.random.randint(step_rng, x.shape,                       
                                                   minval=0, maxval=2).astype(jnp.float32) * 2 - 1

      jvp = jax.jvp(lambda x: drift_fn(x,t_in[:,None]), (x,), (eps,))[1]

      if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        #quite an unstable calculation for vesde
        dsigma_dt=jax.vmap(grad_sigma_t)(t_in)[:,None]
        d2sigma_dt2 = jax.vmap(jax.jacfwd(lambda T: grad_sigma_t(T)))(t_in)[:,None]
        dt = -jnp.sum(jvp * eps, axis=tuple(range(1, len(x.shape))))[:,None] - (1.0 / sigma_t(t_in))[:,None] * dsigma_dt +   d2sigma_dt2/jnp.abs(dsigma_dt)
      else:
        dsigma_dt=jax.vmap(grad_sigma_t)(t_in)[:,None]
        dt =-jnp.sum(jvp * eps, axis=tuple(range(1, len(x.shape))))[:,None]#-1e3*dsigma_dt
      grad_xt=jnp.concatenate([grad,dt],axis=-1)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, grad_xt.shape)
      grad_norm = jnp.linalg.norm(
        grad_xt.reshape((grad_xt.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      
      xt=jnp.concatenate([x,t_in[:,None]],axis=-1)
      xt_mean = xt + batch_mul(step_size, grad_xt)
      xt = xt_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, xt[:,:-1], xt_mean[:,:-1],xt[:,-1],xt_mean[:,-1]

    _, x, x_mean,t,t_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x,t,t))
    return x, x_mean,t,t_mean









@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, rng, x, t):
    return x, x


def shared_predictor_update_fn(rng, state, x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(rng, x, t)


def shared_corrector_update_fn(rng, state, x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(rng, x, t)


def get_pc_sampler(sde, model, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples as well as
    the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)
  def pc_sampler(rng, state):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples, number of function evaluations
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t
      rng, step_rng = random.split(rng)
      x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    # Denoising is equivalent to running one predictor step without adding noise.
    return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return jax.pmap(pc_sampler, axis_name='batch')


def get_ode_sampler(sde, model, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.
  this is different as it uses diffrax instead of scipy
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

  Returns:
    A sampling function that takes random states, and a replicated training state and returns samples
    as well as the number of function evaluations during sampling.
  """
  if(method != 'RK45'):
    raise ValueError('get_ode_sampler: method is not RK45')

  @jax.pmap
  def denoise_update_fn(rng, state, x):
    score_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = jnp.ones((x.shape[0],)) * eps
    _, x = predictor_obj.update_fn(rng, x, vec_eps)
    return x

  @jax.pmap
  def drift_fn(state, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(prng, pstate, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      prng: An array of random state. The leading dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      z: If present, generate samples from latent code `z`.
    Returns:
      Samples, and the number of function evaluations.
    """
    # Initial sample
    rng = flax.jax_utils.unreplicate(prng)
    rng, step_rng = random.split(rng)
    if z is None:
      # If not represent, sample the latent code from the prior distibution of the SDE.
      x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
    else:
      x = z

    solver = diffrax.Tsit5()  # Equivalent to RK45
    t0 = sde.T
    t1 = eps
    dt0 = t1-t0 # Automatic initial step size
    saveat = diffrax.SaveAt(ts=[t1])  # Save only final time 

    def ode_func(t, x,args):
      x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
      vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
      drift = drift_fn(pstate, x, vec_t)
      return to_flattened_numpy(drift) 
    
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)


    # we integrate from eps to T
    # this is the opposite of the true thing
    term=diffrax.ODETerm(ode_func)
    solution = diffrax.diffeqsolve(
                    term,
                    solver,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                    y0=to_flattened_numpy(x),  # Keep as JAX array (no NumPy conversion)
                    saveat=saveat,
                    stepsize_controller=stepsize_controller # No step limit
                )
    nfe = solution.stats["num_steps"]  # Closest analog to nfev
    x = solution.ys[0].reshape((jax.local_device_count(),) + shape)

    # this may not be the best integrator to use!
    # Extract results
    """   
    solution = integrate.solve_ivp(
                (ode_func),
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
        
    x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)
    nfe = solution.nfev
    """
    # Denoising is equivalent to running one predictor step without adding noise
    if denoise:
      rng, *step_rng = random.split(rng, jax.local_device_count() + 1)
      step_rng = jnp.asarray(step_rng)
      x = denoise_update_fn(step_rng, pstate, x)

    x = inverse_scaler(x)
    return x, nfe

  return ode_sampler


#### Joint sampling ------------------------------------------------------------------------------------------



def get_joint_update(config,sde, model, shape, inverse_scaler,scaler, eps,integration_method='exact',hutchinson_type = 'Rademacher'): 
  
  """Create a joint sampler based on the predictor-corrector (PC) sampler.

  """
  denoise=config.sampling.noise_removal
  predictor_warmup = get_predictor(config.sampling.predictor.lower())
  #TODO have these as different correctors and predictors
  corrector_joint = get_corrector('langevin_time_exact') 
  corrector=get_corrector(config.sampling.corrector.lower())
  # Create predictor & corrector update functions
  predictor_update_fn_final = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor_warmup,
                                          probability_flow=config.sampling.probability_flow,
                                          continuous=config.training.continuous)


  corrector_update_fn_final = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=config.training.continuous,
                                          snr=config.sampling.snr,
                                          n_steps=config.sampling.n_steps_joint)
  
  corrector_update_joint = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector_joint,
                                          continuous=config.training.continuous,
                                          snr=config.sampling.snr,
                                          n_steps=config.sampling.n_steps_joint)

  def joint_update(prng, pstate,samples):
    """ The jint update.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples, number of function evaluations
    """

    #prng
    rng = (prng)
    # Initial sample
    x = scaler(samples)
    #initial timesteps
    timesteps = jnp.linspace(sde.T, eps, sde.N)
    t=eps
    vec_t_initial = jnp.ones(shape[0]) * t
    shape_vmap=(1,shape[1])
    rng, step_rng = random.split(rng) 
    n_samples=config.sampling.per_chain_samples

    #update algorithm

    def joint_update(carry,i):
      rng, x, x_mean, vec_t,x_list,x_mean_list,vec_t_list = carry
      rng, step_rng = random.split(rng)
      x_new, x_mean_new,v_t_new,v_t_mean_new =corrector_update_joint(step_rng, pstate, x, vec_t)
      v_t_new, v_t_mean_new = jnp.clip(v_t_new, eps, sde.T), jnp.clip(v_t_mean_new, eps, sde.T)
      rng, step_rng = random.split(rng)     

      x_list = x_list.at[i].set(x_new)
      x_mean_list = x_mean_list.at[i].set(x_mean_new)
      vec_t_list = vec_t_list.at[i].set(v_t_new)
      return (rng, x_new, x_mean_new, v_t_new,x_list,x_mean_list,vec_t_list), None

    #set the list
    x_list = jnp.zeros((n_samples,shape[0], shape[1]))   # Shape (batch_size, dims, n_samples)
    x_mean_list = jnp.zeros((n_samples,shape[0], shape[1])) 
    vec_t_list = jnp.zeros((n_samples,shape[0])) 
    
    carry=rng, x, x, vec_t_initial,x_list,x_mean_list,vec_t_list
    final_carry, _ = jax.lax.scan(joint_update, carry, jnp.arange(n_samples))
    rng, x, x_mean, vec_t,x_list,x_mean_list,vec_t_list = final_carry

    x,x_mean,vec_t=x_list.reshape(-1,shape[-1]), x_mean_list.reshape(-1,shape[-1]), vec_t_list.flatten()
    timesteps = jax.vmap(lambda t: jnp.linspace(t, eps, sde.N // 100))(vec_t).T
    #warmup stage
    def loop_body_final(i, val):
      rng, x, x_mean = val
      vec_t = timesteps[i] #this needs to be made a vector!

      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn_final(step_rng, pstate, x, vec_t) # use the same predictor
      rng, step_rng = random.split(rng)
      x, x_mean=corrector_update_fn_final(step_rng, pstate, x, vec_t)
      return rng, x, x_mean

    

    #rng, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body_final, (rng, x, x_mean))
    return jnp.concatenate([vec_t[:,None],inverse_scaler(x_mean if denoise else x)],-1), sde.N * (config.sampling.n_steps_each+config.sampling.per_chain_samples*config.sampling.n_steps_joint + 1)

  return jax.pmap(joint_update, axis_name='batch')






#### Metropolis sampling ------------------------------------------------------------------------------------------


def get_metropolis_update(config,sde, model,model_noise, shape, inverse_scaler,scaler, eps,integration_method='exact',hutchinson_type = 'Rademacher'): 
  
  """Create a metropolis sampler based on the predictor-corrector (PC) sampler.

  """
  denoise=config.sampling.noise_removal
  predictor_warmup = get_predictor(config.sampling.predictor.lower())
  #TODO have these as different correctors and predictors
  corrector = get_corrector(config.sampling.corrector.lower()) 
  # Create predictor & corrector update functions
  predictor_update_fn_warmpup = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor_warmup,
                                          probability_flow=config.sampling.probability_flow,
                                          continuous=config.training.continuous)

  
  corrector_update_fn_metropolis = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=config.training.continuous,
                                          snr=config.sampling.snr,
                                          n_steps=config.sampling.n_steps_metropolis)
  #this is different at the moment because in future we will have other distributions
  def noise_foward(x,noise_state):
    variables = {'params': noise_state.params, **noise_state.model_state}
    return model_noise.apply(variables, x, train=False, mutable=False)
 

  #@jax.pmap
  def drift_fn(state, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)

    return (rsde.sde)(x, t)[0]
  if(integration_method=='estimate'):
    #@jax.pmap
    def div_fn(state, x, t, eps):
      """Pmapped divergence of the drift function."""
      div_fn = get_div_fn(lambda x, t: drift_fn(state, x, t))
      return div_fn(x, t, eps)
    if hutchinson_type == 'Gaussian':
      rng, step_rng = random.split(rng)
      epsilon = jax.random.normal(step_rng, shape)
    elif hutchinson_type == 'Rademacher':
      rng, step_rng = random.split(rng)
      epsilon = jax.random.randint(step_rng, shape,                          
                                        minval=0, maxval=2).astype(jnp.float32) * 2 - 1
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
  elif(integration_method=='exact'):
    epsilon=None
    #@jax.pmap
    def div_fn(state, x, t,eps):
      """Pmapped divergence of the drift function."""
      div_fn = get_div_fn_exact(lambda x, t: drift_fn(state, x, t))
      return div_fn(x, t, eps)
  else:
    raise NotImplementedError(f"Method of integration {integration_method} unknown.")     

  def metropolis_update(prng, pstate,pstate_noise,samples):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples, number of function evaluations
    """

    #prng
    rng = (prng)

    # Initial sample
    x = scaler(samples)
    #initial timesteps
    timesteps = jnp.linspace(sde.T, eps, sde.N)
    t=eps
    
    vec_t_initial = jnp.ones(shape[0]) * t
    shape_vmap=(1,shape[1])

    ######### Integration setup!
    solver = diffrax.Tsit5() 
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

    def ode_func(t, x,args):
    # x should be shape (n_dims,)
      # t should be shape ()
        sample = x.reshape(shape_vmap)
        vec_t = jnp.ones(shape_vmap[0]) * t
        drift = drift_fn(pstate, sample, vec_t)
        div=div_fn(pstate, sample, vec_t, epsilon)
        prior_sigma=jnp.log(sde.marginal_prob(jnp.zeros_like(x), vec_t)[1]) #p(sigma) propto 1/sigma
  
        drift = mutils.to_flattened_numpy(drift)
        logp_grad = mutils.to_flattened_numpy(div)

        z = mutils.from_flattened_numpy(drift, shape_vmap)
        delta_logp = logp_grad.sum()
        prior_logp = (sde.prior_logp)(z)
        l_hood_new = (prior_logp + delta_logp)-prior_sigma 
        return jnp.concatenate([l_hood_new,l_hood_new],axis=0)  # Concatenate to match the shape of x

    def ode_func(t, x,args):
      sample = x[:-shape_vmap[0] * shape_vmap[1]].reshape(shape_vmap)
      vec_t =  jnp.ones(shape_vmap[0]) * t
      drift = drift_fn(pstate, sample, vec_t)
      logp_grad=div_fn(pstate, sample, vec_t, epsilon)
      drift = mutils.to_flattened_numpy(drift)
      logp_grad = mutils.to_flattened_numpy(logp_grad)

      return jnp.concatenate([drift, logp_grad], axis=0)
     


    term=diffrax.ODETerm(ode_func)

    def solve_ode(inputs):
        t0, x_batch = inputs[0], inputs[1:]
        init=jnp.concatenate([mutils.to_flattened_numpy(x_batch), jnp.zeros((shape_vmap[1],))], axis=0)
        t1=sde.T
        dt0 = t1 - t0 # Automatic initial step size
        saveat = diffrax.SaveAt(ts=[t1])  # Save only final time 
        solution = diffrax.diffeqsolve(
          term,
          solver,
          t0=t0,
          t1=t1,
          dt0=dt0,
          y0=init,
          saveat=saveat,
          stepsize_controller=stepsize_controller
        )
        zp = jax.lax.stop_gradient(jnp.asarray(solution.ys[0]))
        z = mutils.from_flattened_numpy(zp[:-shape_vmap[0] * shape_vmap[1]], shape_vmap)
        delta_logp = (zp[-shape_vmap[0] * shape_vmap[1]:].reshape((shape_vmap[0], shape_vmap[1]))).sum(-1)
        prior_logp = sde.prior_logp(z)
 
        prior_sigma=jnp.log(sde.marginal_prob(jnp.zeros_like(x), jnp.ones(shape_vmap[0])*t0)[1]) #p(sigma) propto 1/sigma

        likelihood_value = (prior_logp + delta_logp)#-prior_sigma ##-prior_sigma / jnp.log(2) #we dont use bits/dim here
        #technically the likelihood is off by a constant from the inv scalar, but we dont care about that because we only care about difference of log likelihood
        include_scaler_trasnform=False
        if(include_scaler_trasnform==True):
          likelihood_value = likelihood_value + jnp.log(jnp.abs(jnp.linalg.det(jax.grad(inverse_scaler)(z))))


        return likelihood_value  # Stop gradient to avoid backprop through ODE solver

    ############
    ### Get the likelihood of the original samples
    ################
      
    
    # fist step, evaluate the likelihood of the original points
    batched_solve = eqx.filter_vmap(solve_ode)
    inputs_intial=jnp.concatenate([vec_t_initial.reshape(-1,1),x],axis=1)
    solutions_initial = batched_solve(inputs_intial)
    l_hood_inital = jnp.asarray(solutions_initial)[:,0]
    
    
    rng, step_rng = random.split(rng) 

    ###### metropolis step
    ######### Metropolis loop!
    n_samples=config.sampling.per_chain_samples
 

    def loop_body_metropolis(carry,i):
      rng, x, x_mean, vec_t, l_hood,x_list,x_mean_list,vec_t_list = carry
      rng, step_rng = random.split(rng)

      v_t_new=div_fn(pstate, x, vec_t, epsilon)
      rng, step_rng = random.split(rng)
      x_new, x_mean_new =corrector_update_fn_metropolis(step_rng, pstate, x, v_t_new)
      rng, step_rng = random.split(rng)     
    
      #batched_solve = eqx.filter_vmap(solve_ode)
      #inputs=jnp.concatenate([v_t_new.reshape(-1,1),x_new],axis=1)
      #solutions = batched_solve(inputs)
      #l_hood_new = jnp.asarray(solutions)[:,0]
      l_hood_new=l_hood
      
      # Metropolis acceptance
      accept = 1#jnp.exp(l_hood_new - l_hood)
      rng, step_rng = random.split(rng)
      u = random.uniform(step_rng, (shape[0],))
      mask = u <=accept
      x_new = jnp.where(mask[:, None], x_new, x)
      x_mean_new=jnp.where(mask[:, None], x_mean_new, x_mean)
      l_hood = jnp.where(mask, l_hood_new, l_hood)
      v_t_new = jnp.where(mask, v_t_new, vec_t)
      x_list = x_list.at[i].set(x_new)
      x_mean_list = x_mean_list.at[i].set(x_mean_new)
      vec_t_list = vec_t_list.at[i].set(v_t_new)
      return (rng, x_new, x_mean_new, v_t_new, l_hood,x_list,x_mean_list,vec_t_list), None

    x_list = jnp.zeros((n_samples,shape[0], shape[1]))   # Shape (batch_size, dims, n_samples)
    x_mean_list = jnp.zeros((n_samples,shape[0], shape[1])) 
    vec_t_list = jnp.zeros((n_samples,shape[0])) 
    
    carry=rng, x, x, vec_t_initial, l_hood_inital,x_list,x_mean_list,vec_t_list
    final_carry, _ = jax.lax.scan(loop_body_metropolis, carry, jnp.arange(n_samples))
    rng, x, x_mean, vec_t, l_hood,x_list,x_mean_list,vec_t_list = final_carry

    x,x_mean,vec_t=x_list.reshape(-1,shape[-1]), x_mean_list.reshape(-1,shape[-1]), vec_t_list.flatten()
    timesteps = jax.vmap(lambda t: jnp.linspace(t, eps, sde.N // 100))(vec_t).T
    #warmup stage
    def loop_body_final(i, val):
      rng, x, x_mean = val
      vec_t = timesteps[i] #this needs to be made a vector!

      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn_warmpup(step_rng, pstate, x, vec_t) # use the same predictor
      rng, step_rng = random.split(rng)
      x, x_mean=corrector_update_fn_metropolis(step_rng, pstate, x, vec_t)
      return rng, x, x_mean
    # Metropolis sampling

    

    #rng, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body_final, (rng, x, x_mean))
    return jnp.concatenate([vec_t[:,None],inverse_scaler(x_mean if denoise else x)],-1), sde.N * (config.sampling.n_steps_each+config.sampling.per_chain_samples*config.sampling.n_steps_metropolis + 1)

  return jax.pmap(metropolis_update, axis_name='batch')



##################
#### utils for the metropolis

def get_div_fn(fn):
  """Create the divergence function of `fn`x using the Hutchinson-Skilling trace estimator."""
  ## Forward-mode differentiation (faster)
  def div_fn(x, t, eps):
      jvp = jax.jvp(lambda x: fn(x, t), (x,), (eps,))[1]
      return jnp.sum(jvp * eps, axis=tuple(range(1, len(x.shape))))

  return div_fn

def get_div_fn_exact(fn):
    """Create the exact divergence function of `fn`, where `fn: R^n -> R^n`."""   
    def div_fn(x, t,eps=None):

        # Compute per-sample Jacobian, ensuring differentiation is only w.r.t x

        jacobian = jax.vmap(jax.jacfwd(fn,argnums=0))(x, t[:,None])  # Differentiate w.r.t. x only
        
        return jnp.diagonal(jacobian, axis1=-2, axis2=-1)  # Return the diagonal along the last two axes

    return div_fn

