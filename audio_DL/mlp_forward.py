import numpy as np


class MLP(object):

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.weights:
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)

        return activations


    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))

        return y

if __name__ == "__main__":
    mlp = MLP()

    inputs = np.random.rand(mlp.num_inputs)
    output = mlp.forward_propagate(inputs)

    print("Network activation: {}".format(output))