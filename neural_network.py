from typing import Iterable

import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:
    """
    Applies the logistic function to an array elementwise
    :param x: an input array
    :return: An array of the same size
    """
    return 1.0/(1.0+np.exp(-1*x))


def logistic_prime(x: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the logistic function to an array elementwise
    :param x: an input array
    :return: An array of the same size
    """
    normal_logistic = 1.0/(1.0+np.exp(-1*x))
    return normal_logistic * (1-normal_logistic)


class MLP:
    def __init__(self, layer_sizes: Iterable[int]):
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a feedforward run of the network and returns the output
        :param x: An array of equal size to the first layer in the MLP
        :return: An array of equial size to the last layer in the MLP
        """
        activation = x
        for weight, bias in zip(self.weights, self.biases):
            activation = logistic((weight @ activation) + bias)
        return activation

    def backprop(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Runs backprop for a single example thorugh the network
        :param x: An input
        :param y: A target output of the value
        :return: The weight and bias rates of change with respect to the cost
        """
        activations = [x]
        zs = []
        delta_weights = []
        delta_biases = []
        for weight, bias in zip(self.weights, self.biases):
            zs.append((weight @ activations[-1]) + bias)
            activations.append(logistic(activations[-1]))
        delta_a = activations[-1] - y
        for i in range(1, len(self.weights) - 1):
            delta_z = delta_a * logistic_prime(activations[-i])
            delta_biases = delta_z
            delta_weights = np.outer(delta_z @ activations[-i - 1])
            delta_a = self.weights[-i].T @ delta_z
        delta_weights.reverse()
        delta_biases.reverse()
        return delta_weights, delta_biases
