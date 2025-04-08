
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
import equinox
from typing import Any
import matplotlib.pyplot as plt
import math
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import simple_score
from models import noise_class
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import likelihood
import sde_lib
from absl import flags
from models import nutils as nmutils
FLAGS = flags.FLAGS
OVERWRITE_CHECKPOINT=True

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)
    

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
  optimizer, optimize_fn = losses.get_optimizer(config)
  state = mutils.State(step=0, opt_state=optimizer.init(initial_params), lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,params=initial_params,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds, eval_ds= datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)


  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                  config.data.vector_dim)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())
  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    try:
        batch_data = next(train_iter)
    except StopIteration:
        # Restart the iterator when the dataset is exhausted.

        train_iter = iter(train_ds)
        batch_data = next(train_iter)
    batch = jax.tree_util.tree_map(
                lambda x: scaler(x._numpy()), batch_data
            )
    #batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)

    next_rng = jnp.asarray(next_rng)
    # Execute one training step


    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss))
      writer.scalar("training_loss", loss, step)
      flat_params = flax.traverse_util.flatten_dict(pstate.params)

      for key, value in flat_params.items():
          tag = "weights/" + "/".join(key)
          # Convert the JAX array to a NumPy array before logging.
          writer.histogram(tag, np.asarray(value), step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      try:
          eval_data = next(eval_iter)
      except StopIteration:
          eval_iter = iter(eval_ds)
          eval_data = next(eval_iter)
      eval_batch = jax.tree_util.tree_map(
                lambda x: scaler(x._numpy()), eval_data
            )
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      if jax.host_id() == 0:
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
        writer.scalar("eval_loss", eval_loss, step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf,overwrite=OVERWRITE_CHECKPOINT)

      # Generate and save samples
      # Generate and save samples (vector data)
      if config.training.snapshot_sampling:
          rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
          sample, n = sampling_fn(sample_rng, pstate)
          this_sample_dir = os.path.join(sample_dir, f"iter_{step}_host_{jax.host_id()}")
          tf.io.gfile.makedirs(this_sample_dir)
            # Save the raw vector samples
          with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.npz"), "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, sample=sample)
              fout.write(io_buffer.getvalue())
              if(step>0):
                if jax.host_id() == 0:
                  writer.histogram("posterior_samples/dim_0", sample.reshape(-1,sample.shape[-1])[:, 0], step)
                  writer.histogram("posterior_samples/dim_1", ((sample.reshape(-1,sample.shape[-1])[:, 1])),step)







"""Evaluation for score-based generative models (vector data version)."""



def evaluate(config, workdir, eval_folder="eval"):
  """Evaluate trained models for vector data (no image metrics)."""

  # Create evaluation directory.
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline.
  # Here additional_dim is 1 (or you can adjust if needed) and uniform dequantization is off.
  train_ds, eval_ds= datasets.get_dataset(
      config,
      additional_dim=1,
      uniform_dequantization=config.data.uniform_dequantization,
      evaluation=True)
  
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer, optimize_fn = losses.get_optimizer(config)
  state = mutils.State(step=0, opt_state=optimizer.init(initial_params), lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,params=initial_params,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args
  
  # initialize noise model 
  rng, step_rng = jax.random.split(rng)
  noise_model, init_noise_model_state, initial_params_noise = nmutils.init_noise_model(step_rng, config)
  optimizer_noise, optimize_fn_noise = losses.get_optimizer(config)
  state_noise = nmutils.NoiseState(step=0, opt_state=optimizer_noise.init(initial_params_noise), lr=config.optim.lr,
                       model_state=init_noise_model_state,
                       ema_rate=config.model.ema_rate,params=initial_params_noise,
                       params_ema=initial_params_noise,
                       rng=rng)  # pytype: disable=wrong-keyword-args


  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_dir_noise = os.path.join(workdir, "checkpoints_noise")

  # Setup SDE.
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                         N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                           N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  #samlping parameters and data for likelihood evaluation
  num_sampling_rounds = int(
        math.ceil(config.eval.num_samples / config.eval.batch_size) ) 
  train_ds_bpd, eval_ds_bpd = datasets.get_dataset(config,
                                                      additional_dim=None,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1

  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5

  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  num_bpd_rounds=len(ds_bpd) * bpd_num_repeats



  # Build the one-step evaluation function if loss computation is enabled.
  if config.eval.enable_loss:
    optimizer,optimize_fn = losses.get_optimizer(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, score_model,
                                   train=False,
                                   optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step),
                            axis_name='batch', donate_argnums=1)


  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler
                                                 ,how=config.eval.integration_method,hutchinson_type=config.eval.hutchinson)#,num_repeats=5 if config.eval.bpd_dataset.lower() == "test" else 1,)

  # Build the sampling function.
  if config.eval.enable_sampling:
    # For vectors, sampling shape is [batch, vector_dim]
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.vector_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Create a simple metadata container for resuming evaluation.
  @flax.struct.dataclass
  class EvalMeta:
    ckpt_id: int
    sampling_round_id: int
    bpd_round_id: int
    rng: Any

  # For simplicity, we define these based on the evaluation dataset size.
 # For vector data, you might not repeat over multiple batches for likelihood.

  # Restore evaluation meta state (if available).
  eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, sampling_round_id=-1, bpd_round_id=-1, rng=rng)
  eval_meta = checkpoints.restore_checkpoint(eval_dir, eval_meta, step=None, prefix=f"meta_")
  bpd_round_id=eval_meta.bpd_round_id
  # Set starting checkpoint and round indices.
  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
    begin_sampling_round = 0
  elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = num_bpd_rounds
    begin_sampling_round = eval_meta.sampling_round_id + 1
  else:
    begin_ckpt = eval_meta.ckpt_id #+ 1
    begin_bpd_round = 0
    begin_sampling_round = 0

  rng = eval_meta.rng

  logging.info("Starting evaluation from checkpoint: %d" % (begin_ckpt,))

  # Evaluate each checkpoint.
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}")
    while not tf.io.gfile.exists(ckpt_filename):
      logging.warning("Waiting for checkpoint_%d ..." % (ckpt,))
      time.sleep(60)

    # Try restoring checkpoint (with retries in case of file issues).
    for _ in range(3):
      try:
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        break
      except Exception:
        time.sleep(60)
    pstate = flax.jax_utils.replicate(state)


    ckpt_filename_noise = os.path.join(checkpoint_dir_noise, f"checkpoint_{ckpt}")
    while not tf.io.gfile.exists(ckpt_filename_noise):
      logging.warning("Waiting for checkpoint_%d ..." % (ckpt,))
      time.sleep(60)

    for _ in range(3):
      try:
        state_noise = checkpoints.restore_checkpoint(checkpoint_dir_noise, state_noise, step=ckpt)
        break
      except Exception:
        time.sleep(60)
    pstate_noise = flax.jax_utils.replicate(state_noise)

    # Evaluate loss on the full evaluation dataset (if enabled).
    if config.eval.enable_loss:
      all_losses = []
      
      eval_iter = iter(eval_ds)
      for i, batch in enumerate(eval_iter):
        eval_batch =jax.tree_util.tree_map(
                lambda x: scaler(x._numpy()), batch
            )
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        (_, _), p_eval_loss = p_eval_step((next_rng, pstate), eval_batch)
        loss = jnp.mean(flax.jax_utils.unreplicate(p_eval_loss))
        all_losses.append(loss)
        if (i + 1) % 100 == 0:
          logging.info("Checkpoint %d: finished %d batches for loss evaluation." % (ckpt, i + 1))
      all_losses = jnp.asarray(all_losses)
      loss_save_path = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
      with tf.io.gfile.GFile(loss_save_path, "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=float(all_losses.mean()))
        fout.write(io_buffer.getvalue())


    # (Optional) Likelihood evaluation for vectors.
    if config.eval.enable_bpd:
      bpds = []
      begin_repeat_id = begin_bpd_round // len(ds_bpd)
      begin_batch_id = begin_bpd_round % len(ds_bpd)
      # Repeat multiple times to reduce variance when needed
      for repeat in range(begin_repeat_id, bpd_num_repeats):
        bpd_iter = iter(ds_bpd)

        for _ in range(begin_batch_id):
          next(bpd_iter)  # pytype: disable=wrong-arg-types
        for batch_id in range(begin_batch_id, len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
          rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
          step_rng = jnp.asarray(step_rng)
          bpd = likelihood_fn(step_rng, pstate, eval_batch['joint_data'])[0]
          bpd = bpd.reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                  "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, bpd)
              fout.write(io_buffer.getvalue())

          eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
          # Save intermediate states to resume evaluation after pre-emption
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
            keep=1,
            prefix=f"meta_{jax.host_id()}_")
    else:
      # Skip likelihood computation and save intermediate states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
      try:
        checkpoints.save_checkpoint(
          eval_dir,
          eval_meta,
          step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1,
          keep=1,
          prefix=f"meta_{jax.host_id()}_")
      except Exception as e:
        print(f"An error occurred while saving the checkpoint: {e}")
    if(config.sampling.joint==True):
      joint_update = sampling.get_joint_update(config=config, sde=sde, model=score_model,shape=sampling_shape,
                                                         inverse_scaler= inverse_scaler, scaler=scaler,eps=sampling_eps)

    # Sampling: generate and save vector samples.
    if config.eval.enable_sampling:
      state = jax.device_put(state)
      state_noise = jax.device_put(state_noise)

      for r in range(begin_sampling_round, num_sampling_rounds):
        if jax.host_id() == 0:
          logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
        rng, *sample_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)

        if(config.sampling.joint==False):
          samples, _ =(sampling_fn)(sample_rng, pstate)
          # Save raw vector samples
        elif(config.sampling.joint==True): 
          samples, _ =(sampling_fn)(sample_rng, pstate)
          rng, *sample_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
          samples, _ =joint_update(sample_rng,pstate,samples)
        else:
          raise NotImplementedError(f"Sampling joint {config.sampling.joint} not T/F.")
        
        
        
        this_sample_dir = os.path.join(
            eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
        tf.io.gfile.makedirs(this_sample_dir)

        sample_save_path = os.path.join(this_sample_dir, f"samples_{r}.npz")
        with tf.io.gfile.GFile(sample_save_path, "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())
        logging.info("Checkpoint %d: saved sampling round %d." % (ckpt, r))

        def clear_memory(x):
          del x
          return None

        samples = jax.tree_map(clear_memory, samples)



        # Update meta state.
        eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
        try:
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
            keep=1,
            prefix=f"meta_{jax.host_id()}_")
        except Exception as e:
          print(f"An error occurred while saving the checkpoint: {e}")
    else:
      # Skip sampling and save intermediate evaluation states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
        keep=1,
        prefix=f"meta_{jax.host_id()}_")


    # Reset round indices for next checkpoint.
    begin_bpd_round = 0
    begin_sampling_round = 0

  meta_files = tf.io.gfile.glob(
    os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
  for file in meta_files:
      try:
          tf.io.gfile.remove(file)
      except tf.errors.PermissionDeniedError as e:
          print(f"Permission denied: {e}")
      except Exception as e:
          print(f"An error occurred: {e}")









##################
### noise run
##################


def noise_train(config, workdir):
  """Runs the training pipeline for the noise model.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """



  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard_noise")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)
    

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  noise_model, init_model_state, initial_params = nmutils.init_noise_model(step_rng, config)
  optimizer, optimize_fn = losses.get_optimizer(config)
  state = nmutils.NoiseState(step=0, opt_state=optimizer.init(initial_params), lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,params=initial_params,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints_noise")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints_noise-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds, eval_ds= datasets.get_dataset_noise(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)


  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler_noise(config)
  inverse_scaler = datasets.get_data_inverse_scaler_noise(config)


  # Build one-step training and evaluation functions
  train_step_fn = losses.noise_get_step_fn(noise_model, train=True, optimize_fn=optimize_fn)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.noise_get_step_fn(noise_model, train=False, optimize_fn=optimize_fn)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)


  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())
  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    try:
        batch_data = next(train_iter)
    except StopIteration:
        # Restart the iterator when the dataset is exhausted.

        train_iter = iter(train_ds)
        batch_data = next(train_iter)
    batch = jax.tree_util.tree_map(
                lambda x: scaler(x._numpy()), batch_data
            )
    #batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)

    next_rng = jnp.asarray(next_rng)
    # Execute one training step


    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss))
      writer.scalar("training_loss", loss, step)
      flat_params = flax.traverse_util.flatten_dict(pstate.params)

      for key, value in flat_params.items():
          tag = "weights/" + "/".join(key)
          # Convert the JAX array to a NumPy array before logging.
          writer.histogram(tag, np.asarray(value), step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      try:
          eval_data = next(eval_iter)
      except StopIteration:
          eval_iter = iter(eval_ds)
          eval_data = next(eval_iter)
      eval_batch = jax.tree_util.tree_map(
                lambda x: scaler(x._numpy()), eval_data
            )
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      if jax.host_id() == 0:
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
        writer.scalar("eval_loss", eval_loss, step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf,overwrite=OVERWRITE_CHECKPOINT)



