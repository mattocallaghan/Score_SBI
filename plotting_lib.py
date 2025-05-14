
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
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
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import likelihood
import sde_lib
from absl import flags
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import glob
from models import nutils as nmutils
FLAGS = flags.FLAGS
OVERWRITE_CHECKPOINT=True



def score_plot(config, workdir, eval_folder="eval"):
  """Evaluate trained models for vector data (no image metrics)."""

  # Create evaluation directory.
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline.
  # Here additional_dim is 1 (or you can adjust if needed) and uniform dequantization is off.

  
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

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDE.
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                         N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                           N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                        N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


  train_ds_bpd, eval_ds_bpd = datasets.get_dataset_plotting(config,
                                                      additional_dim=None,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1

  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 1

  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  num_bpd_rounds=len(ds_bpd) * bpd_num_repeats


  if True:
    likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler
                                                 ,how=config.eval.integration_method,hutchinson_type=config.eval.hutchinson,eps=1e-5)#,num_repeats=5 if config.eval.bpd_dataset.lower() == "test" else 1,)

  # Build the sampling function.


  # Create a simple metadata container for resuming evaluation.
  @flax.struct.dataclass
  class EvalMeta:
    ckpt_id: int
    bpd_round_id: int
    rng: Any



  # Restore evaluation meta state (if available).
  eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, bpd_round_id=-1, rng=rng)
  eval_meta = checkpoints.restore_checkpoint(eval_dir, eval_meta, step=None, prefix=f"meta_")
  bpd_round_id=eval_meta.bpd_round_id
  # Set starting checkpoint and round indices.
  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
  else:
    begin_ckpt = eval_meta.ckpt_id 
    begin_bpd_round = 0

  rng = eval_meta.rng

  logging.info("Starting evaluation from checkpoint: %d" % (begin_ckpt,))

  # Evaluate each checkpoint.
  print('here')

  for ckpt in range(begin_ckpt, config.eval.end_ckpt+1):
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

    # Likelihood evaluation for vectors.
    if True:
      bpds = []
      ds=[]
      gradients=[]
      begin_repeat_id = 0
      begin_batch_id = 1
      # Repeat multiple times to reduce variance when needed
      for repeat in range(0, 5):
        bpd_iter = iter(ds_bpd)

        for _ in range(begin_batch_id):
          next(bpd_iter)  # pytype: disable=wrong-arg-types
        for batch_id in range(begin_batch_id, len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
          rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
          step_rng = jnp.asarray(step_rng)

          bpd = likelihood_fn(step_rng, pstate, eval_batch['joint_data'])[0]
          

          #gradients
          def score_forward(x,labels,state):
            variables = {'params': state.params_ema, **state.model_state}
            return score_model.apply(variables, x, labels, train=False, mutable=False)
          p_score_foward= jax.pmap(score_forward)

          grd = jnp.stack([p_score_foward(eval_batch['joint_data'], epsilon * jnp.ones(eval_batch['joint_data'].shape[:-1]),pstate) for epsilon in np.linspace(config.model.sigma_min,config.model.sigma_max,4)],axis=-1)
          bpds.extend(bpd.T)
          ds.extend(batch['joint_data']._numpy().reshape(-1,batch['joint_data']._numpy().shape[-1]))
          gradients.extend(grd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                  "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, bpd)
              fout.write(io_buffer.getvalue())
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_data_{bpd_round_id}.npz"),
                                  "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, batch['joint_data']._numpy())
              fout.write(io_buffer.getvalue())
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_gradients_{bpd_round_id}.npz"),
                                  "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, grd)
              fout.write(io_buffer.getvalue())

          eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
          # Save intermediate states to resume evaluation after pre-emption
          try:
            checkpoints.save_checkpoint(
              eval_dir,
              eval_meta,
              step=ckpt * (0 + num_bpd_rounds) + bpd_round_id,
              keep=1,
              prefix=f"plot_meta_{jax.host_id()}_")
          except Exception as e:
            logging.error(f"Error when saving intermediate states: {e}")
          
    if True:
      file_paths_likelihood = sorted(glob.glob(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd*")))
      file_paths_data = sorted(glob.glob(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_data*")))
      
      file_paths_gradients = sorted(glob.glob(os.path.join(eval_dir,
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_gradients*")))

      # Load and stack data samples
      data_samples = [np.load(fp)[np.load(fp).files[0]] for fp in file_paths_data[0:]]
      stacked_data = np.vstack(data_samples).reshape(-1,config.data.vector_dim)

      # Load and stack likelihood samples
      likelihood_samples = [np.load(fp)[np.load(fp).files[0]] for fp in file_paths_likelihood[0:]]
      stacked_likelihood = np.vstack(likelihood_samples).flatten()
      gradients_samples = [np.load(fp)[np.load(fp).files[0]] for fp in file_paths_gradients[0:]]

      # Example data
      x = stacked_data[:, 0]
      y = stacked_data[:, 1]
      z = stacked_likelihood

      # Create grid values first.

      # Plot
      fig, axs = plt.subplots(2, 3, figsize=(18, 12))

      # Scatter plot
      scatter = axs[0, 0].scatter(x, y, c=z, cmap='viridis', marker='o')
      fig.colorbar(scatter, ax=axs[0, 0], label='Likelihood')
      axs[0, 0].set_title('Original Data Likelihood')
      axs[0, 0].set_xlabel('X')
      axs[0, 0].set_ylabel('Y')
      axs[0, 0].grid(True)

      # Quiver plots for each epsilon value
      epsilons = np.linspace(config.model.sigma_min,config.model.sigma_max,4)
      for i, epsilon in enumerate(epsilons):
        gradients_samples_temp = [grad_s[:,:,:,i] for grad_s in gradients_samples]

        stacked_gradients = np.vstack(gradients_samples_temp).reshape(-1, config.data.vector_dim)#*epsilon
        
        grad_magnitude = np.linalg.norm(stacked_gradients, axis=-1)
        #grad_magnitude=50
        # Clip gradient magnitudes to avoid outliers
        clip_percentile = 100  # Clip anything above the 99th percentile
        max_grad = np.percentile(grad_magnitude, clip_percentile)

        # Normalize gradients with clipping
        grad_magnitude = np.clip(grad_magnitude, 0, max_grad)
        u = stacked_gradients[:, 0]/grad_magnitude
        v = stacked_gradients[:, 1]/grad_magnitude
        row = (i + 1) // 3
        col = (i + 1) % 3
        # Quiver plot for gradients
        axs[row, col].quiver(x, y, u, v, color='black', scale=30, width=0.002, alpha=0.8)
        max_grad_idx = np.argmax(grad_magnitude)
        axs[row, col].set_title(f'Grads $\\sigma$={epsilon:.2f}, Max Grad: {grad_magnitude[max_grad_idx]:.2f}')
        axs[row, col].set_xlabel('X')
        axs[row, col].set_ylabel('Y')
        axs[row, col].grid(True)

        # Plot the magnitude of maximum gradient
        
        # Mark the point with maximum gradient
        
        axs[row, col].set_xlabel('X')
        axs[row, col].set_ylabel('Y')
        axs[row, col].grid(True)

      # Hide the unused subplot
      axs[1, 2].axis('off')

      # Save the plot
      plt.savefig(os.path.join(eval_dir,
        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_likelihood_gradients_plot.png"), dpi=300)
      plt.show()
      plt.close()
