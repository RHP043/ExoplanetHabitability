import numpy as np


# The Layer, FCLayer, Activation, and Network classes go here
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        pass

    def back_prop(self, outputError, learningRate):
        pass


class FCLayer(Layer):
    def __init__(self, inputSize, output_size):
        self.weights = np.random.rand(inputSize, output_size) - 0.5
        self.bias = np.random.rand(output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias.reshape(1, -1)  # wx + b

        return self.output

    def back_prop(self, outputError, learningRate):
        input_error = np.dot(outputError, self.weights.T)  # dE/dX = dE/dY^T
        weights_error = np.dot(self.input.T, outputError)  # dE dW= X^T dE/dY
        bias = outputError  # df/dB = 1

        self.weights = self.weights - learningRate * weights_error  # w = w - r dE/dB
        self.bias = self.bias - learningRate * bias  # b = b - r dE/dB
        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def back_prop(self, outputError, learningRate):
        return self.activation_prime(self.input) * outputError


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

        self.weights_history = []
        self.biases_history = []

    def add(self, i):
        self.layers.append(i)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict_class(self, input_d, minX, maxX):
        input_denormalized = (input_d * (maxX - minX)) + minX
        # print(input_denormalized)
        input_data = np.array(input_denormalized).reshape(1, 1)
        output = self.predict(input_data)[0]
        predicted_class = f"True ({output})" if output >= 0.5 else f"False ({output})"
        return predicted_class

    def predict(self, input_data):
        len_data = len(input_data)
        result = []
        for i in range(len_data):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, lr):
        len_tr = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(len_tr):
                output = x_train[j]
                for layer in self.layers:
                    # print(j, "1. ", output, layer)
                    output = layer.forward_propagation(output)
                    # print(j, "2. ", output, layer)
                err += self.loss(y_train[j], output)
                errorr = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    errorr = layer.back_prop(errorr, lr)
            err /= len_tr
            print(f"Epoch: {i + 1}/{len_tr}\t\t\t\t\t\tError: {err}")


# functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def binary_crossentropy(y_true, y_pred):
    # y_true is habitability column
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return loss


def binary_crossentropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
