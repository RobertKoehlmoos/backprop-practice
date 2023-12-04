from typing import Sequence

import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:  # (B, C) -> (B -> C)
    """
    Applies the logistic function to an array
    :param x: an input array
    :return: An array of the same size
    """
    return 1.0 / (1.0 + np.exp(-1 * x))


def logistic_prime(x: np.ndarray) -> np.ndarray:  # (B, C) -> (B ->C)
    """
    Applies the derivative of the logistic function to an array
    :param x: an input array
    :return: An array of the same size
    """
    normal_logistic = 1.0 / (1.0 + np.exp(-1 * x))  # efficiency
    return normal_logistic * (1 - normal_logistic)


class MLP:
    # this MLP handles things similarly to the other one, but uses a fully batched approach to improve performance
    def __init__(self, layer_sizes: Sequence[int]):
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y) for y in layer_sizes[1:]]
        # x is the size of the output from the previous layer, y is the size of the layer the weight outputs are going to
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, x: np.ndarray) -> np.ndarray:  # (B, input_layer_size) -> (B, output_layer_size)
        """
        Performs a feedforward run of the network and returns the output
        :param x: An array of equal size to the first layer in the MLP, with shape (B, I)
        :return: An array of equal size to the last layer in the MLP, with shape (B, O)
        """
        activation = x  # (B, I)
        for weight, bias in zip(self.weights, self.biases):
            activation = logistic((activation @ weight.T) + bias)  # (B, previous layer size) -> (B, current layer size)
        return activation

    def sgd(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
            test_y: np.ndarray, epochs: int, batch_size: int, learning_rate: float = 0.2) -> None:
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
        batches = tuple((train_x[j: j + batch_size, :], train_y[j: j + batch_size, :])
                        for j in range(0, len(train_x), batch_size))
        for i in range(epochs):
            for x, y in batches:
                delta_weights, delta_biases = self.backprop(x, y)
                # performing the updates
                # we need to use x.shape[0] to account for the final batch being a potentially different size
                self.weights = [weight - (learning_rate / x.shape[0]) * weight_update.sum(0)
                                for weight, weight_update in zip(self.weights, delta_weights)]
                self.biases = [bias - (learning_rate / x.shape[0]) * bias_update.sum(0)
                               for bias, bias_update in zip(self.biases, delta_biases)]
            # running tests to track performance based on accuracy
            prediction = self.feedforward(test_x)  # (B, o), for predictions and y
            correct = np.sum(np.argmax(prediction, 1) == np.argmax(test_y, 1))
            print(f"Accuracy for epoch {i}: {correct / len(test_x)}")

    def backprop(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Runs backprop for a single example through the network
        :param x: An input with shape (B, I), where I is the size of the first layer
        :param y: A target output of the input with shape (B, O), where O is the size of the last layer
        :return: The weight and bias rates of change with respect to the cost averaged over the full batch
        """
        activations = [x]  # list[(B, I)] to start each entry will be (B, l), where l is the size of the current layer
        zs = []
        delta_weights = []  # list of (B, l, l-1)
        delta_biases = []  # list of (B, l)
        # feedforward to collect activations as the neuron sums (zs)
        for weight, bias in zip(self.weights,
                                self.biases):  # weight (l, l-1), bias (l) where l is the size of the current layer
            # for the first layer I = l-1, so for each layer we do (B, l-1) @ (l, l-1).T = (B, l)
            zs.append(activations[-1] @ weight.T + bias)
            activations.append(logistic(zs[-1]))
        # backprop step
        delta_a = activations[-1] - y  # taking the derivative of the cost for each activation
        delta_z = delta_a * logistic_prime(activations[-1])  # back propagating from the last output to the last sum
        delta_biases.append(delta_z)  # the differentials for biases are equal to the delta for the sum
        # for weights we need (B, L, 1) (B, 1, L-1) @ (B, L, L-1)
        delta_weights.append(
            delta_z[:, :, np.newaxis] @ activations[-2][:, np.newaxis, :])  # calculating differentials for weights
        # continuing to backpropagation through each layer
        for i in range(2, len(self.weights) + 1):
            delta_a = delta_z @ self.weights[-i + 1]  # (B, l + 1) @ (l+1, l).T = (B, l)
            delta_z = delta_a * logistic_prime(activations[-i])  # backpropagation from the activation to the sum
            delta_biases.append(delta_z)
            delta_weights.append(delta_z[:, :, np.newaxis] @ activations[-i - 1][:, np.newaxis,
                                                             :])  # (B, l, 1) @ (B, 1, l-1) = (B, l, l-1)
        delta_weights.reverse()
        delta_biases.reverse()
        return delta_weights, delta_biases
