import numpy as np
import jax.numpy as jnp

class NN():
    
    def __init__(self, activation) -> None:
        super().__init__()
        self.activation = activation

    def gen_params(self, input_size, hiddens_layers: list, output_size, range_w):
        layers = [input_size] + hiddens_layers + [output_size]
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(range_w[0], range_w[1], (sender, receiver))
            biases = np.random.uniform(range_w[0], range_w[1], (1, receiver))
            sender = receiver
            params.append([weights, biases])
        return params
    
    def act_func(self, z: float):
        if self.activation == "relu": return jnp.maximum(0, z)
        elif self.activation == "tanh": return jnp.tanh(z)
        elif self.activation == "sigmoid": return 1 / (1 + jnp.exp(-z))
        else: raise NotImplementedError("Activation function not implemented")