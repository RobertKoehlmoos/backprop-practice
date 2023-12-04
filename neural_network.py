from typing import Sequence, List, Optional

import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:
    """
    Applies the logistic function to an array elementwise
    :param x: an input array
    :return: An array of the same size
    """
    return 1.0 / (1.0 + np.exp(-1 * x))


def logistic_prime(x: np.ndarray) -> np.ndarray:
    """
    Applies the derivative of the logistic function to an array elementwise
    :param x: an input array
    :return: An array of the same size
    """
    normal_logistic = 1.0 / (1.0 + np.exp(-1 * x))  # efficiency
    return normal_logistic * (1 - normal_logistic)


class MLP:
    def __init__(self, layer_sizes: Sequence[int]):
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        # x is the size of the output from the previous layer, y is the size of the layer the weight outputs are going to
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a feedforward run of the network and returns the output
        :param x: An array of equal size to the first layer in the MLP, with shape (n, 1)
        :return: An array of equal size to the last layer in the MLP, with shape (m, 1)
        """
        activation = x
        for weight, bias in zip(self.weights, self.biases):
            activation = logistic((weight @ activation) + bias)
        return activation

    def sgd(self, train_x: List[np.ndarray], train_y: List[np.ndarray], test_x: List[np.ndarray],
            test_y: List[np.ndarray], epochs: int, batch_size: int, learning_rate: float = 0.2) -> None:
        """
        Performs stochastic gradient descent using training data, while
        :param train_x: Training inputs
        :param train_y: Training target
        :param test_x: Testing inputs
        :param test_y: Testing targets
        :param epochs: The number of training runs to perform
        :param batch_size: Size of each training batch before update
        :param learning_rate: Learning rate to use when updating the model's parameters
        """
        for i in range(epochs):
            # training
            batches = (zip(train_x[j: j + batch_size], train_y[j: j + batch_size])
                       for j in range(0, len(train_x), batch_size))
            for batch in batches:
                # creating 0 filled matrices to accumulate the rates of change
                weights_update = [np.zeros(weight.shape) for weight in self.weights]
                biases_update = [np.zeros(bias.shape) for bias in self.biases]
                for x, y in batch:
                    delta_weights, delta_biases = self.backprop(x, y)
                    weights_update = [weight_update + delta_weight
                                      for weight_update, delta_weight in zip(weights_update, delta_weights)]
                    biases_update = [bias_update + delta_bias
                                     for bias_update, delta_bias in zip(biases_update, delta_biases)]
                # performing the updates
                self.weights = [weight - (learning_rate / batch_size) * weight_update
                                for weight, weight_update in zip(self.weights, weights_update)]
                self.biases = [bias - (learning_rate / batch_size) * bias_update
                               for bias, bias_update in zip(self.biases, biases_update)]
            # running tests to track performance based on accuracy
            correct = 0
            for x, y in zip(test_x, test_y):
                prediction = self.feedforward(x)
                if np.argmax(prediction) == np.argmax(y):
                    correct += 1
            print(f"Accuracy for epoch {i}: {correct / len(train_x)}")

    def backprop(self, x: np.ndarray, y: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
        """
        Runs backprop for a single example through the network
        :param x: An input with shape (n, 1) where n is the size of the first layer
        :param y: A target output of the input with shape (m, 1), where m is the size of the last layer
        :return: The weight and bias rates of change with respect to the cost
        """
        activations = [x]
        zs = []
        delta_weights = []
        delta_biases = []
        # feedforward to collect activations as the neuron sums (zs)
        for weight, bias in zip(self.weights, self.biases):
            zs.append((weight @ activations[-1]) + bias)
            activations.append(logistic(zs[-1]))
        # backprop step
        delta_a = activations[-1] - y  # taking the derivative of the cost for each activation
        delta_z = delta_a * logistic_prime(activations[-1])  # back propagating from the last output to the last sum
        delta_biases.append(delta_z)  # the differentials for biases are equal to the delta for the sum
        delta_weights.append(delta_z @ activations[-2].T)  # calculating differentials for weights
        # continuing to backpropagation through each layer
        for i in range(2, len(self.weights) + 1):
            delta_a = self.weights[
                          -i + 1].T @ delta_z  # backpropagation from the layer's sum to the previous layer's activations
            delta_z = delta_a * logistic_prime(activations[-i])  # backpropagation from the activation to the sum
            delta_biases.append(delta_z)
            delta_weights.append(delta_z @ activations[-i - 1].T)
        delta_weights.reverse()
        delta_biases.reverse()
        return delta_weights, delta_biases
