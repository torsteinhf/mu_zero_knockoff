
# serve as a distributer between representation, dynamics and prediction

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax

from config import (
    MAX_LOG_TILE, LOOKBACK, INPUT_DIM, ROLL_AHEAD,
    SEQ_DIM, ABSTRACT_DIM, NUM_ACTIONS, LEARNING_RATE
)
from nn import NNr, NNd, NNp

### state encoding helper functions ###
def encode_state(state) -> np.ndarray:
    flat = np.array(state, dtype=np.float32).flatten()
    return np.where(flat > 0, np.log2(np.maximum(flat, 1.0)) / MAX_LOG_TILE, 0.0)

def build_state_seq(encoded_history: list) -> np.ndarray:
    blank = np.zeros(INPUT_DIM, dtype=np.float32)
    n = len(encoded_history)
    need = LOOKBACK + 1
    frames = [
        encoded_history[n - need + i] if (n - need + 1) >= 0 else blank
        for i in range(need)
    ]
    return np.concat(frames) # has shape (SEQ_DIM,)
#######################################


class NNManager():

    def __init__(self, seed: int = 0) -> None:
        rng_key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(rng_key, 3)

        self.nnr = NNr()
        self.nnd = NNd()
        self.nnp = NNp()

        self.params = {
            "r" : self.nnr.init(k1, jnp.zeros(SEQ_DIM)),
            "d" : self.nnd.init(k2, jnp.zeros(ABSTRACT_DIM), jnp.zeros(NUM_ACTIONS)),
            "p" : self.nnp.init(k3, jnp.zeros(NUM_ACTIONS))
        }

        self.optimizer = optax.adam(LEARNING_RATE)
        self.opt_state = self.optimizer.init(self.params)

        self._nnr_func = jax.jit(self.nnr.apply)
        self._nnd_func = jax.jit(self.nnd.apply)
        self._nnp_func = jax.jit(self.nnp.apply)

        self._train_func = self._build_train_func()

    def represent(self, state_seq) -> jnp.ndarray:
        return self._nnr_func(self.params["r]"], jnp.array(state_seq, dtype = jnp.float32))

    def dynamics(self, abstract: jnp.ndarray, acton_idx: int):
        action_ohe = jax.nn.one_hot(acton_idx, NUM_ACTIONS)
        return self._nnd_func(self.params["d"], abstract, action_ohe)

    def predict(self, abstract: jnp.ndarray):
        return self._nnp_func(self.params["p"], abstract)

    # One BPTT training step
    def train_step(self, batch: list) -> float:
        ss = jnp.array([s["state_seq"] for s in batch], dtype=jnp.float32)
        a = jnp.array([s["actions"] for s in batch], dtype=jnp.float32)
        tp = jnp.array([s["policies"] for s in batch], dtype=jnp.float32)
        tv = jnp.array([s["values"] for s in batch], dtype=jnp.float32)
        tr = jnp.array([s["rewards"] for s in batch], dtype=jnp.float32)

        self.params, self.opt_state, loss = self._train_func(
            self.params, self.opt_state, ss, a, tp, tv, tr
        )

        return float(loss)

    def _build_train_func(self): # returns a jit func that performs one grad step
        nnr = self.nnr
        nnd = self.nnd
        nnp = self.nnp
        optimizer = self.optimizer
        
        def loss_func(params, ss, a, tp, tv, tr): # vmap BPTT loss over batch
            # t for target - policy, value, reward
            def single(ss_i, a_i, tp_i, tv_i, tr_i): # i sub for iterating over batch in vmap
                abstract = nnr.apply(params["r"], ss_i)
                p, v = nnp.apply(params["p"], abstract) # init prediction of abstract state
                L = (
                    -jnp.dot(tp_i[0], jnp.log(p + 1e-8))    # policy CE w/ 
                    + (v - tv[0]) ** 2                      # value MSE
                )
                for t in range(ROLL_AHEAD):
                    a_ohe = jax.nn.one_hot(a[t], NUM_ACTIONS)
                    abstract, r = nnd.apply(params["d"], abstract, a_ohe)
                    p, v = nnp.apply(params["p"], abstract)
                    L += (
                        -jnp.dot(tp_i[t+1], jnp.log(p + 1e-8))
                        + (v - tv_i[t + 1]) ** 2
                        + (r - tr_i[t + 1]) ** 2 
                    )
                return L
            return jnp.mean(jax.vmap(single)(ss, a, tp, tv, tr))
        
        @jax.jit
        def train_step(params, opt_state, ss, a, tp, tv, tr):
            loss, grads = jax.value_and_grad(loss_func)(params, ss, a, tp, tv, tr)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        return train_step

    def save_params(self, filename = "params.pkl"):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self.params, f)
        print(f"Params saved to {filename}")

    def load_params(self, filename = "params.pkl"):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
        self.opt_state = self.optimizer.init(self.params)
        print(f"Params loaded from {filename}")