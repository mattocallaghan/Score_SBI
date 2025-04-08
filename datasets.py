import jax
import tensorflow as tf
import os
import numpy as np
import sbibm
import pickle
import jax.numpy as jnp
from sklearn.datasets import make_moons
from models.nutils import sample_joint
def get_data_scaler(config):
    """Data normalizer using saved mean and std, or identity if standardization is disabled."""
    if not config.data.standardize:
        # Return identity function if standardization is disabled
        return lambda x: x

    # Build the path to the stats file
    dataset_name = config.data.dataset
    num_sim = config.data.num_simulations
    data_dir = config.data.data_path
    stats_filename = f"{dataset_name}_{num_sim}_stats.npy"
    stats_path = os.path.join(data_dir, stats_filename)
    
    # Check if the stats file exists
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. Generate the dataset first to compute mean and std."
        )
    
    # Load the stats

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)  # Load saved stats if they exist
    mean, std = stats['mean'], stats['std']

    # Define the scaler function using lambda
    return lambda x: (x - mean) / std

def get_data_inverse_scaler(config):
    """Inverse data normalizer using saved mean and std, or identity if standardization is disabled."""
    if not config.data.standardize:
        # Return identity function if standardization is disabled
        return lambda x: x

    # Build the path to the stats file
    dataset_name = config.data.dataset
    num_sim = config.data.num_simulations
    data_dir = config.data.data_path
    stats_filename = f"{dataset_name}_{num_sim}_stats.npy"
    stats_path = os.path.join(data_dir, stats_filename)
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. Generate the dataset first."
        )
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)  # Load saved stats if they exist
    mean, std = stats['mean'], stats['std']

    
    # Define the inverse scaler function using lambda
    return lambda x: x * std + mean



def get_dataset(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
    """Load a dataset based on config settings."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size // jax.device_count()
    num_epochs = None if not evaluation else 1
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size] if additional_dim else [jax.local_device_count(), per_device_batch_size]
    dataset_name = config.data.dataset

    train_ds, eval_ds = load_custom_dataset(dataset_name, config)
    
    train_ds = preprocess_and_batch(train_ds, batch_dims, config, evaluation, uniform_dequantization)
    eval_ds = preprocess_and_batch(eval_ds, batch_dims, config, evaluation, uniform_dequantization)
    
    return train_ds, eval_ds

def load_custom_dataset(dataset_name, config):
    """Load custom dataset from external files or generate it if needed."""
    data_path = os.path.join(config.data.data_path, f"{dataset_name}_{config.data.num_simulations}.npy")
    stats_path = os.path.join(config.data.data_path, f"{dataset_name}_{config.data.num_simulations}_stats.npy")

    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)  # Load saved stats if they exist
        mean, std = stats['mean'], stats['std']
    else:
        if(config.data.benchmark==True):
            data = generate_sbibm_data(config)
            np.save(data_path, data)
            
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            
            # Save mean and std to a file
            with open(stats_path, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)
        else:
            data = generate_other_data(config)
            np.save(data_path, data)
            
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            
            # Save mean and std to a file
            with open(stats_path, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)

   
    dataset = tf.data.Dataset.from_tensor_slices({"joint_data": data[:, :]})

    # Normalize the dataset (if desired) by subtracting mean and dividing by std
    
    # Split dataset into training and evaluation sets
    train_ds = dataset.take(int(0.8 * len(data)))
    eval_ds = dataset.skip(int(0.8 * len(data)))

    return train_ds, eval_ds


def generate_sbibm_data(config):
    """Generate data using SBIBM task and simulator, with conditional prior sampling."""
    task = sbibm.get_task(config.data.task)
    
    if config.data.task_prior:  # Use task prior if the config specifies so
        prior = task.get_prior()
        simulator = task.get_simulator()
        theta = prior(num_samples=config.data.num_simulations)
        data = simulator(theta).numpy()
    
    else:  # Define a custom prior or uniform prior if task_prior is False
        # For simplicity, define a uniform prior over a custom range for theta
        reference_samples = np.concatenate([task.get_reference_posterior_samples(num_observation=_).numpy() for _ in range(1, 11)])
        
        low_bounds = jax.nn.relu(reference_samples.min(0) - 0.05)
        high_bounds = jax.nn.relu(reference_samples.max(0) + 0.05)
        theta = np.random.uniform(low=low_bounds, high=high_bounds, size=(config.data.num_simulations, 4))        
        data = task.get_simulator()(theta).numpy()  # Simulate data based on sampled theta
    
    # Concatenate theta and data for the final dataset
    return np.concatenate((theta, data), axis=1)


def generate_other_data(config):
    """Generate data using other task and simulator, with conditional prior sampling."""
    other_tasks=['make_moons']
    assert config.data.benchmark==False, "Benchmark should be set to False for other tasks."
    assert config.data.dataset in other_tasks, f"Dataset name should be one of {other_tasks}"
    
    if(config.data.dataset=='make_moons'):
        data = make_moons(n_samples=config.data.num_simulations, noise=0.1)[0]

    # Concatenate theta and data for the final dataset
    return data

def preprocess_and_batch(dataset, batch_dims, config, evaluation, uniform_dequantization):
    """Preprocess dataset, apply augmentations, and batch it."""
    def preprocess_fn(d):
        # no preprocessing for now
        return d
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      dataset = dataset.batch(batch_size, drop_remainder=True)
    #dataset = dataset.batch(batch_dims[-1], drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=10000)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


#############
# NOISE
#############


def get_data_scaler_noise(config):
    """Data normalizer using saved mean and std, or identity if standardization is disabled."""
    return lambda x: x


def get_data_inverse_scaler_noise(config):
    """Inverse data normalizer using saved mean and std, or identity if standardization is disabled."""

    return lambda x: x


def get_dataset_noise(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
    """Load a dataset based on config settings."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size // jax.device_count()
    num_epochs = None if not evaluation else 1
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size] if additional_dim else [jax.local_device_count(), per_device_batch_size]
    dataset_name = config.data.dataset

    train_ds, eval_ds = load_custom_dataset_noise(dataset_name, config)
    
    train_ds = preprocess_and_batch(train_ds, batch_dims, config, evaluation, uniform_dequantization)
    eval_ds = preprocess_and_batch(eval_ds, batch_dims, config, evaluation, uniform_dequantization)
    
    return train_ds, eval_ds

def load_custom_dataset_noise(dataset_name, config):
    """Load custom dataset from external files or generate it if needed."""
    data_path = os.path.join(config.data.data_path, f"{dataset_name}_{config.data.num_simulations}.npy")
    stats_path = os.path.join(config.data.data_path, f"{dataset_name}_{config.data.num_simulations}_stats.npy")

    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)  # Load saved stats if they exist
        mean, std = stats['mean'], stats['std']
    else:
        if(config.data.benchmark==True):
            data = generate_sbibm_data(config)
            np.save(data_path, data)
            
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            
            # Save mean and std to a file
            with open(stats_path, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)
        else:
            data = generate_other_data(config)
            np.save(data_path, data)
            
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            
            # Save mean and std to a file
            with open(stats_path, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)

    
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, data.shape[0])
    sample_joint_vmap = jax.vmap(lambda d, r: sample_joint(d, config.model.sigma_max, config.model.sigma_min, r, jnp.linspace(1.0, 1e-5,config.model.num_scales)))
    x, y = sample_joint_vmap(data, rngs)
    dataset = tf.data.Dataset.from_tensor_slices({"data": x,'labels':y})
    
    # Split dataset into training and evaluation sets
    train_ds = dataset.take(int(0.8 * len(data)))
    eval_ds = dataset.skip(int(0.8 * len(data)))

    return train_ds, eval_ds



############################################
###### plotting dataset ####################
############################################






def get_dataset_plotting(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
    """Load a dataset based on config settings."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size // jax.device_count()
    num_epochs = None if not evaluation else 1
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size] if additional_dim else [jax.local_device_count(), per_device_batch_size]
    dataset_name = config.data.dataset

    train_ds, eval_ds = load_custom_dataset_plotting(dataset_name, config)
    
    train_ds = preprocess_and_batch(train_ds, batch_dims, config, evaluation, uniform_dequantization)
    eval_ds = preprocess_and_batch(eval_ds, batch_dims, config, evaluation, uniform_dequantization)
    
    return train_ds, eval_ds

def load_custom_dataset_plotting(dataset_name, config):
    """Load custom dataset from external files or generate it if needed."""
    dimension = config.data.vector_dim
    bound = config.model.sigma_max
    grid_point=config.eval.grid_points
    
    # Generate uniform data within the specified bounds
    #data = np.random.uniform(low=-bound, high=bound, size=(100000, dimension))
    # Generate meshgrid data within the specified bounds
    grid_points = np.linspace(-2, 2, int((grid_point)))
    mesh = np.meshgrid(*[grid_points,] * dimension)
    data = np.stack([mesh[_].flatten() for _ in range(len(mesh))],-1)

    
    # Create a TensorFlow dataset from the generated data
    dataset = tf.data.Dataset.from_tensor_slices({"joint_data": data})
     
    # Split dataset into training and evaluation sets
    #train_ds = dataset.take(int(0.8 * len(data)))
    #eval_ds = dataset.skip(int(0.8 * len(data)))
    train_ds= eval_ds = dataset
    return train_ds, eval_ds