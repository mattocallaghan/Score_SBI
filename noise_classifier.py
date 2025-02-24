import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
from optax import adam, apply_updates
import math
import os
import orbax.checkpoint
from flax.training import checkpoints, train_state
import flax
from flax.training import orbax_utils
from sklearn.datasets import make_moons
import optax
T=200 #number of time points

rng = jax.random.PRNGKey(0)

######################
# Model Definition   #
######################

class Noise_Classifier_Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(10*T)(x))
        x = nn.relu(nn.Dense(5*T)(x))
        x = nn.Dense(T)(x)
        return x

@jax.jit
def sample_joint(x, sigma_max, sigma_min=0.01,rng=jax.random.PRNGKey(42)):
    rng2,rng3 = jax.random.split(rng,2)
    ts=jnp.linspace(1.0, 1e-3, T)
    ss = sigma_min * (sigma_max / sigma_min) ** ts
    y = jax.random.randint(rng2, (x.shape[0],), 0, T)

    x = x + ss[y].reshape(-1,1) * jax.random.normal(rng3, x.shape)
    return x,y


######################
# Training the model #
######################

if __name__ == '__main__':
    load = 0
    ckpt_dir = './logs'
    s_epoch = 5
    n_epoch = 1000
    bs = 100
    learning_rate = 1e-3

    train_X, _ = make_moons(n_samples=100000, noise=0.1)
    train_X = jnp.array(train_X)

    n_iter = math.ceil(train_X.shape[0] / bs)

    net=Noise_Classifier_Network()
    variables=net.init(rng, train_X[:1])
    optimizer = adam(learning_rate=learning_rate)
    ######################
    # Checkpointing      #
    ######################


    state = train_state.TrainState.create(
        apply_fn=net.apply,
        params=variables['params'],
        tx=optimizer)
    ckpt = {'model': state}


    ######################
    # loss function      #
    ######################
    from operator import getitem
    import pickle

    @jax.jit
    def crit(logits, labels):
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    ######################
    # Training step      #
    ######################

    @jax.jit
    def train_step(state: train_state.TrainState, x,y):
        """Train for a single step."""

        @jax.jit
        def loss_fn(params):
            outputs=jax.vmap(net.apply,in_axes=(None, 0))({'params': params}, x)
            return jax.vmap(crit)(outputs, y).mean() #this is the mean over the batch, is that right for bce
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss


    epoch = 0
    ######################
    # training loop      #
    ######################
    sigma_max=10.0
    sigma_min=0.01
    for epoch in range(epoch, n_epoch):
        epoch=epoch+1
        epoch_loss = 0
        for i in range(n_iter):
            rng, rng_key = jax.random.split(rng) #update the rng key based on the last one
            x,y=(sample_joint)(train_X[i*bs:(i+1)*bs],sigma_max,sigma_min,rng=rng_key)

            state, loss=train_step(state, x, y)
            epoch_loss += loss
        with open('./logs/model_params.pkl', 'wb') as f:
            pickle.dump(state.params, f)
        epoch_loss = epoch_loss / n_iter
        print(f'Epoch {epoch}, Loss {epoch_loss}')
        ckpt = {'model': state}


    params=state.params
    x,y=(sample_joint)(train_X[:],sigma_max,sigma_min)
    outputs=nn.softmax(jax.vmap(net.apply,in_axes=(None, 0))({'params': params}, x))
    import matplotlib.pyplot as plt
    plt.title('Predicted vs True')
    plt.scatter(jnp.argmax(outputs,1),y)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.hist(y)
    plt.show()

