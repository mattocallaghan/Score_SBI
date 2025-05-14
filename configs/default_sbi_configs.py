from ml_collections import ConfigDict
import ml_collections

def get_config():
    config = ConfigDict()

    # Global seed.
    config.seed = 42

    # Training settings.
    config.training = ConfigDict()
    config.training.batch_size = 64
    config.training.n_iters = 25000           # Total number of training iterations.
    config.training.n_jitted_steps = 5         # Number of steps jitted together.
    config.training.log_freq = 20        # Log frequency (must be divisible by n_jitted_steps).
    config.training.snapshot_freq_for_preemption = 20
    config.training.eval_freq = 20
    config.training.snapshot_freq = 1000
    config.training.sde = 'vesde'               # Options: 'vpsde', 'vesde', or 'subvpsde'.
    config.training.continuous = True
    config.training.reduce_mean = True
    config.training.likelihood_weighting = True
    config.training.snapshot_sampling = True

    # Evaluation settings.
    config.eval = ConfigDict()
    config.eval.enable_loss = True
    config.eval.bpd_dataset = 'test'            # Use either 'train' or 'test'.
    config.eval.enable_bpd = False #likelihoods not ready yet
    config.eval.enable_sampling = True
    config.eval.num_samples = 10000
    config.eval.batch_size = 5000
    config.eval.begin_ckpt = 0                  # Starting checkpoint index.
    config.eval.end_ckpt = 1000              # Ending checkpoint index.

    # Data settings.
    config.data = ConfigDict()
    config.data.dataset = 'gaussian_linear'             
    config.data.num_simulations = 10000         # For custom datasets (if applicable).
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.uniform_dequantization = False #not used
    # For sbibm-based tasks (if used):
    config.data.task_prior = True
    config.data.standardize=True
    config.data.benchmark=False
    # Model settings.
    config.model = ConfigDict()
    config.model.ema_rate = 0.999
    config.model.num_scales = 500              # Number of discretization steps for the SDE.
    config.model.beta_min = 0.1                 # For VPSDE / subVPSDE.
    config.model.beta_max = 20.0                # For VPSDE / subVPSDE.
    config.model.sigma_min = 0.01               # For VESDE.
    config.model.sigma_max = 3.0               # For VESDE.

    # Optimizer settings.
    config.optim = ConfigDict()
    config.optim.lr = 2e-4
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 0.0
    config.optim.warmup = 1000
    config.optim.grad_clip = 1.0

    # Sampling settings.
    config.sampling = ConfigDict()
    config.sampling.method = 'ode'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.noise_removal = False         # Whether to perform a final denoising step.
    config.sampling.predictor = 'reverse_diffusion'  # Options include 'reverse_diffusion', 'euler_maruyama', etc.
    config.sampling.corrector = 'langevin'       # Options include 'langevin', 'ald', or 'none'.
    config.sampling.snr = 0.16
    config.sampling.n_steps_each = 1
    config.sampling.probability_flow = False
    config.sampling.n_steps_joint=1
    config.sampling.joint=True
    config.sampling.per_chain_samples=1

    config.noise_model = ConfigDict()
    config.noise_model.name='simple_noise'
    config.noise_model.hidden_size=5
    config.noise_model.dropout=0.1
    return config



"""
The snr (signal-to-noise ratio) parameter of LangevinCorrector somewhat behaves like a temperature parameter. 
Larger snr typically results in smoother samples, while smaller snr gives more diverse but lower quality samples.
 Typical values of snr is 0.05 - 0.2, and it requires tuning to strike the sweet spot.
For VE SDEs, we recommend choosing config.model.sigma_max to be
the maximum pairwise distance between data samples in the training dataset.
"""