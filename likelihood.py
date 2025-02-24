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

import jax
import flax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
from models import utils as mutils
import diffrax

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    ## Reverse-mode differentiation (slower)
    # def div_fn(x, t, eps):
    #     grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
    #     grad_fn_eps = jax.grad(grad_fn)(x)
    #     return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

    ## Forward-mode differentiation (faster)
    def div_fn(x, t, eps):
        jvp = jax.jvp(lambda x: fn(x, t), (x,), (eps,))[1]
        return jnp.sum(jvp * eps, axis=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(sde, model, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """

  def drift_fn(state, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  @jax.pmap
  def p_div_fn(state, x, t, eps):
    """Pmapped divergence of the drift function."""
    div_fn = get_div_fn(lambda x, t: drift_fn(state, x, t))
    return div_fn(x, t, eps)

  p_drift_fn = jax.pmap(drift_fn)  # Pmapped drift function of the reverse-time SDE
  p_prior_logp_fn = jax.pmap(sde.prior_logp)  # Pmapped log-PDF of the SDE's prior distribution

  def likelihood_fn(prng, pstate, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      z: A JAX array of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    rng, step_rng = jax.random.split(flax.jax_utils.unreplicate(prng))
    shape = data.shape
    if hutchinson_type == 'Gaussian':
      epsilon = jax.random.normal(step_rng, shape)
    elif hutchinson_type == 'Rademacher':
      epsilon = jax.random.randint(step_rng, shape,
                                   minval=0, maxval=2).astype(jnp.float32) * 2 - 1
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def ode_func(t, x,args):
      sample = mutils.from_flattened_numpy(x[:-shape[0] * shape[1]], shape)
      vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t
      drift = mutils.to_flattened_numpy(p_drift_fn(pstate, sample, vec_t))
      logp_grad = mutils.to_flattened_numpy(p_div_fn(pstate, sample, vec_t, epsilon))
      return jnp.concatenate([drift, logp_grad], axis=0)


    init = jnp.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0)


    solver = diffrax.Tsit5()  # Equivalent to RK45
    t0 = sde.T
    t1 = eps
    dt0 = eps - sde.T # Automatic initial step size
    saveat = diffrax.SaveAt(ts=[t1])  # Save only final time 



    term=diffrax.ODETerm(ode_func)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
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
    # Extract results
    nfe = solution.stats["num_steps"]  # Closest analog to nfev
    zp = jnp.asarray(solution.ys[0])
    z = mutils.from_flattened_numpy(zp[:-shape[0] * shape[1]], shape)
    delta_logp = zp[-shape[0] * shape[1]:].reshape((shape[0], shape[1]))
    prior_logp = p_prior_logp_fn(z)
    bpd = -(prior_logp + delta_logp) #/ jnp.log(2)
    #N = jnp.prod(jnp.array(shape[2:]))
    #bpd = bpd / N
    # A hack to convert log-likelihoods to bits/dim
    # based on the gradient of the inverse data normalizer.
    #offset = jnp.log2(jax.grad(inverse_scaler)(0.)) + 8.
    #bpd += offset
    return bpd, z, nfe



  return likelihood_fn







