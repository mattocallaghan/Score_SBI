from configs.default_sbi_configs import get_config as base_get_config
import ml_collections

"""
The main condition is for the scaling by sigma, so we dont condition but we do scale by sigma
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    
    config.model.name='simple_score'
    
    config.model.nf = 4
    config.model.dropout=0.00
    # Whether to condition on the noise level/time (set to True for a conditional score model).
    config.model.conditional = True  
    # The hidden layer size in the network.
    config.model.hidden_size = 250  
    # The number of fully-connected layers (excluding the output layer).
    config.model.num_layers = 5  
    # Scale factor for Gaussian Fourier features.
    config.model.fourier_scale = 3.0 
    config.model.time_dense_size=2
    # Choose the type of time embedding: 'fourier' or 'positional'.
    config.model.embedding_type = 'fourier'  
    config.model.nonlinearity='swish'
    config.model.scale_by_sigma=True

    # You can also add or override other keys if necessary.
    config.data.dataset = 'make_moons'             
    config.data.num_simulations = 10000         # For custom datasets (if applicable).
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.uniform_dequantization = False #not used
    config.data.vector_dim=2
    # For sbibm-based tasks (if used):
    config.data.benchmark = True       # A task name available in SBIBM.

    config.sampling.noise_removal = False         # Whether to perform a final denoising step.

    config.eval.begin_ckpt = 20                  # Starting checkpoint index.
    config.eval.end_ckpt = 20
    config.eval.grid_points=100
    config.sampling.n_steps_joint=3 ##how many skips in the metropolis step
    config.sampling.per_chain_samples=100000
    config.eval.batch_size=20000
    config.eval.num_samples=1 #this doesnt seem to work without this 
    config.eval.integration_method='exact'
    config.eval.hutchinson='Rademacher'   #need to prepegate this
    config.sampling.joint=False
    config.sampling.method = 'pc'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.optim.weight_decay = 1e-3



    return config