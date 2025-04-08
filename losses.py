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

"""All functions related to loss computation and optimization.
"""
import optax
import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul
from models import nutils

def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer.lower() == "adam":
        if hasattr(config.optim, "linear_decay_steps"):  # for progressive distillation
            stable_training_schedule = optax.linear_schedule(
                init_value=config.optim.lr,
                end_value=0.0,
                transition_steps=config.optim.linear_decay_steps,
            )
        else:
            stable_training_schedule = optax.constant_schedule(config.optim.lr)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0,
                    end_value=config.optim.lr,
                    transition_steps=config.optim.warmup,
                ),
                stable_training_schedule,
            ],
            [config.optim.warmup],
        )

        if not jnp.isinf(config.optim.grad_clip):
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=config.optim.grad_clip),
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.optim.beta1,
                    eps=config.optim.eps,
                    weight_decay=config.optim.weight_decay,
                ),
            )
        else:
            optimizer = optax.adamw(
                learning_rate=schedule,
                b1=config.optim.beta1,
                eps=config.optim.eps,
                weight_decay=config.optim.weight_decay,
            )

    elif config.optim.optimizer.lower() == "radam":
        beta1 = config.optim.beta1
        beta2 = config.optim.beta2
        eps = config.optim.eps
        weight_decay = config.optim.weight_decay
        lr = config.optim.lr
        optimizer = optax.chain(
            optax.scale_by_radam(b1=beta1, b2=beta2, eps=eps),
            optax.add_decayed_weights(weight_decay, None),
            optax.scale(-lr),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    def optimize_fn(grads, opt_state, params):

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return optimizer, optimize_fn






def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    data = batch['joint_data']

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    if not likelihood_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score + batch_mul(z, 1. / std))
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn





def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state

    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.params
      states = state.model_state
      opt_state=state.opt_state

      (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
      grad = jax.lax.pmean(grad, axis_name='batch')
      new_params, new_opt_state = optimize_fn(grad, opt_state, params)
      new_params_ema = jax.tree_util.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_params
      )
      step = state.step + 1
      new_state = state.replace(
                    step=step,
                    params=new_params,
                    params_ema=new_params_ema,
                    model_state=new_model_state,
                    opt_state=new_opt_state,
      )
    else:
      loss, _ = loss_fn(step_rng, state.params, state.model_state, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn



######
######
######    NOISE CLASSIFIER
######
######
    

    ######################
    # Training step      #
    ######################



def get_noise_loss_fn(noise_model, train):
  """Create a loss function for training with arbirary SDEs.

  Args:


  Returns:
    A loss function.
  """

  def noise_loss_fn(params, states, batch, rng):
    """Compute the loss function.

    Args:
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.
      rng: A JAX random state.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """
    rng, step_rng = random.split(rng)
    noise_net = nutils.get_noise_model_fn(noise_model, params, states, train=train)
    data = batch['data']
    labels = batch['labels'][:, 0]
    output, new_model_state = noise_net(data, rng=step_rng)
    log_probs = jax.nn.log_softmax(output)  # Add epsilon for numerical stability
    one_hot_labels = jax.nn.one_hot(labels, num_classes=output.shape[-1])  # Convert labels to one-hot

    loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)  # Per-sample loss
    return jnp.mean(loss), new_model_state  # Mean loss over batch

  return noise_loss_fn





def noise_get_step_fn(noise_model, train, optimize_fn=None):
  """Create a one-step training/evaluation function.

  Args:


  Returns:
    A one-step function for training or evaluation.
  """

  noise_loss_fn = get_noise_loss_fn(noise_model, train)
  def noise_step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state

    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(noise_loss_fn, argnums=0, has_aux=True)
    if train:
      params = state.params
      states = state.model_state
      opt_state=state.opt_state

      (loss, new_model_state), grad = grad_fn(params, states, batch,step_rng)
      grad = jax.lax.pmean(grad, axis_name='batch')
      new_params, new_opt_state = optimize_fn(grad, opt_state, params)
      new_params_ema = jax.tree_util.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_params
      )
      step = state.step + 1
      new_state = state.replace(
                    step=step,
                    params=new_params,
                    params_ema=new_params_ema,
                    model_state=new_model_state,
                    opt_state=new_opt_state,
      )
    else:
      loss, _ = noise_loss_fn(state.params_ema, state.model_state, batch,step_rng)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return noise_step_fn