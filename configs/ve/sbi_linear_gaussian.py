from configs.default_sbi_configs import get_config as base_get_config
import ml_collections

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    
    config.model.name='simple_score'
    config.model.nf = 64  
    config.model.dropout=0.05
    # Whether to condition on the noise level/time (set to True for a conditional score model).
    config.model.conditional = True  
    # The hidden layer size in the network.
    config.model.hidden_size = 10  
    # The number of fully-connected layers (excluding the output layer).
    config.model.num_layers = 4  
    # Scale factor for Gaussian Fourier features.
    config.model.fourier_scale = 16.0  
    # Choose the type of time embedding: 'fourier' or 'positional'.
    config.model.embedding_type = 'fourier'  
    config.model.nonlinearity='relu'
    config.model.scale_by_sigma=True

    # You can also add or override other keys if necessary.
    config.data.dataset = 'gaussian_linear'             
    config.data.num_simulations = 10000         # For custom datasets (if applicable).
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.uniform_dequantization = False #not used
    config.data.vector_dim=20
    # For sbibm-based tasks (if used):
    config.data.task = 'gaussian_linear'        # A task name available in SBIBM.
    config.data.task_prior = True
    config.data.benchmark = True       # A task name available in SBIBM.


    return config