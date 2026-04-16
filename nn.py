import flax.linen as nn
import jax.numpy as jnp
from config import ABSTRACT_DIM, NUM_ACTIONS

# Representation Network : state sequence  -> abstract state s
class NNr (nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x);  x = nn.relu(x)
        x = nn.Dense(128)(x);  x = nn.relu(x)
        x = nn.Dense(ABSTRACT_DIM)(x)
        return x


# Dynamics Network : NNd: (abstract_state, action_onehot) -> (next_abstract_state, reward).
class NNd (nn.Module):

    @nn.compact
    def __call__(self, state, action_oh):
        x = jnp.concatenate([state, action_oh])
        x = nn.Dense(128)(x);  x = nn.relu(x)
        x = nn.Dense(128)(x);  x = nn.relu(x)
        next_state = nn.Dense(ABSTRACT_DIM)(x)
        reward     = nn.Dense(1)(x)[0]
        return next_state, reward


# Prediction Network : NNp: abstract_state -> (policy, value).
class NNp(nn.Module):

    @nn.compact
    def __call__(self, state):
        x      = nn.Dense(64)(state);  x = nn.relu(x)
        policy = nn.softmax(nn.Dense(NUM_ACTIONS)(x))
        value  = nn.Dense(1)(x)[0]
        return policy, value