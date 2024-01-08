from matplotlib import pyplot as plt

from models.base_model import base_classifier
import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function will be used in backprop.
def sigmoid_derivative(x):
    return x * (1 - x)


# relu activation function
def relu(x):
    return np.maximum(0, x)


# Derivative of ReLU function
def relu_derivative(x):
    return 1.0 * (x > 0)


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


def cross_entropy_cost(A2, Y, parameters):
    # number of training example
    m = Y.shape[1]
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters, cache, X, Y):
    # number of training example
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), relu_derivative(A1))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def gradient_descent(parameters, grads, learning_rate=0.01):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


# Neural Network
class Nn(base_classifier):

    def __init__(self, train_data=None, test_data=None):
        super().__init__(train_data, test_data)
        self.classifier = None
        self.val_accuracies = None
        self.val_losses = None
        self.train_losses = None

    def classify(self, is_sgd=False):
        preds = forward_propagation(self.X_train, self.bgd())[0]
        preds[preds > 0.5] = 1
        preds[preds < 0.5] = 0
        self.predictions = preds.reshape(preds.shape[1], ).tolist()

    def sgd(self):
        # optimization with stochastic gradient decent.
        # splits X and y's
        split = int(0.8 * len(self.X_train))
        train_input = self.X_train.values[:split]
        train_output = self.y_train.values[:split]
        val_input = self.X_train.values[split:]
        val_output = self.y_train.values[split:]

        # Initialize weights and biases
        input_size = train_input.shape[1]
        hidden_size = 8
        output_size = 1

        np.random.seed(11)  # for reproducibility
        hidden_weights = np.random.randn(input_size, hidden_size)
        hidden_bias = np.zeros((1, hidden_size))
        output_weights = np.random.randn(hidden_size, output_size)
        output_bias = np.zeros((1, output_size))

        epochs = 5000
        learning_rate = 0.01

        # Initialize lists to store losses and accuracy
        train_losses = []
        val_losses = []
        val_accuracies = []

        def cross_entropy_loss(y_true, y_pred):
            return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Training the neural network
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            correct_predictions = 0
            for i in range(len(train_input)):
                input_point = train_input[i:i + 1]
                output_point = train_output[i:i + 1]

                # Forward propagation
                hidden_activation = relu(np.dot(input_point, hidden_weights) + hidden_bias)
                output_activation = sigmoid(np.dot(hidden_activation, output_weights) + output_bias)

                # Calculate the error
                error = output_point - output_activation

                # Calculate the cross-entropy loss for the training set
                train_loss += cross_entropy_loss(output_point, output_activation)

                # Backpropagation
                output_delta = error * sigmoid_derivative(output_activation)
                hidden_error = output_delta.dot(output_weights.T)
                hidden_delta = hidden_error * relu_derivative(hidden_activation)

                # Update weights and biases using gradient descent
                output_weights += hidden_activation.T.dot(output_delta) * learning_rate
                output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
                hidden_weights += input_point.T.dot(hidden_delta) * learning_rate
                hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

            # Calculate average training loss for the epoch
            train_loss /= len(train_input)

            # Validation during training to calculate accuracy and cross-entropy loss
            for j in range(len(val_input)):
                val_input_point = val_input[j:j + 1]
                val_output_point = val_output[j:j + 1]
                val_hidden_activation = relu(np.dot(val_input_point, hidden_weights) + hidden_bias)
                val_output_activation = sigmoid(np.dot(val_hidden_activation, output_weights) + output_bias)
                # Predicted class is 1 if output_activation > 0.5, else 0
                predicted_class = 1 if val_output_activation > 0.5 else 0
                if predicted_class == val_output[j]:
                    correct_predictions += 1
                # Calculate the cross-entropy loss for the validation set
                val_loss += cross_entropy_loss(val_output_point, val_output_activation)

            # Calculate average validation loss and accuracy for the epoch
            val_loss /= len(val_input)
            val_accuracy = correct_predictions / len(val_input)

            # Append losses and accuracy to the lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            self.train_losses = train_losses
            self.val_losses = val_losses
            self.val_accuracies = val_accuracies

            # Print losses and accuracy every 1000 epochs
            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch}: Validation Accuracy: {val_accuracy}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    def bgd(self):
        X_train = self.X_train.values.T
        y_train = self.y_train.values.reshape(1, self.y_train.shape[0])
        X_test = self.X_test.values.T
        y_test = self.y_test.values.reshape(1, self.y_test.shape[0])

        def define_structure(X, Y):
            input_unit = X.shape[0]  # size of input layer
            hidden_unit = 32  # hidden layer of size 4
            output_unit = Y.shape[0]  # size of output layer
            return input_unit, hidden_unit, output_unit

        (input_unit, hidden_unit, output_unit) = define_structure(X_train, y_train)
        print("The size of the input layer is:  = " + str(input_unit))
        print("The size of the hidden layer is:  = " + str(hidden_unit))
        print("The size of the output layer is:  = " + str(output_unit))

        def parameters_initialization(input_unit, hidden_unit, output_unit):
            np.random.seed(2)
            W1 = np.random.randn(hidden_unit, input_unit) * 0.01
            b1 = np.zeros((hidden_unit, 1))
            W2 = np.random.randn(output_unit, hidden_unit) * 0.01
            b2 = np.zeros((output_unit, 1))
            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}

            return parameters

        # data used in batch gradient decent
        split = int(0.8 * X_train.shape[1])
        train_input = X_train[:, :split]
        train_output = y_train[:, :split]
        val_input = X_train[:, split:]
        val_output = y_train[:, split:]

        def neural_network_model(X, Y, X_val, Y_val, hidden_unit, num_iterations=1000):
            np.random.seed(3)
            input_unit = define_structure(X, Y)[0]
            output_unit = define_structure(X, Y)[2]

            parameters = parameters_initialization(input_unit, hidden_unit, output_unit)

            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']

            # Initialize lists to store losses and accuracy
            train_losses = []
            val_losses = []
            val_accuracies = []

            for i in range(0, num_iterations):
                A2, cache = forward_propagation(X, parameters)
                cost = cross_entropy_cost(A2, Y, parameters)
                grads = backward_propagation(parameters, cache, X, Y)
                parameters = gradient_descent(parameters, grads)

                train_loss = cost
                train_losses.append(train_loss)

                # Validation during training to calculate accuracy and cross-entropy loss
                A2_val, _ = forward_propagation(X_val, parameters)
                val_cost = cross_entropy_cost(A2_val, Y_val, parameters)
                val_loss = val_cost
                val_losses.append(val_loss)

                predictions = (A2_val > 0.5)
                val_accuracy = np.mean(predictions == Y_val)
                val_accuracies.append(val_accuracy)

                self.train_losses = train_losses
                self.val_losses = val_losses
                self.val_accuracies = val_accuracies

                if i % 1000 == 0:
                    print("Cost after iteration %i: %f" % (i, cost))
                    print(f"Validation Accuracy after iteration %i: %f" % (i, val_accuracy))

                return parameters

        return neural_network_model(train_input, train_output, val_input, val_output, 8, num_iterations=15000)

    def plot_loss_accuracy(self):
        # After training, plot the losses and accuracy
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
