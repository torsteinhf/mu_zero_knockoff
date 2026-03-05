import pickle
import jax
import jax.numpy as jnp

class NN():

    def __init__(self, activation) -> None:
        self.activation = activation

    def gen_params(self, num_input, hiddens_layers: list, range_w):
        pass
        # Building the network from a configuration file that specifies details such as the number, 
        # size and type of layers, plus their interconnectivity.
    
    def act_func(self, z: float):
        if self.activation == "relu": return jnp.maximum(0, z)
        elif self.activation == "tanh": return jnp.tanh(z)
        elif self.activation == "sigmoid": return 1 / (1 + jnp.exp(-z))
        else: raise NotImplementedError("Activation function not implemented")

    # ? hvordan løse ?
    def preproc(self):
        pass
    def postproc(self):
        pass
    def run_forward(self, params):
        pass
    def train(self, params): # data & target
        pass


    def save_params(self, params, filename):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(params, f)

    def load_params(self, params, filename):
        with open(filename + ".pkl", 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data