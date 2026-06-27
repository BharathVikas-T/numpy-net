import numpy as np


class NeuralNetwork:
    """
    3-layer neural network built from scratch using only NumPy.
    Architecture: 784 (input) -> 128 (hidden, ReLU) -> 10 (output, Softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr  # learning rate

        # --------------- WEIGHT INITIALIZATION ---------------
        # W1: connects input layer (784) to hidden layer (128)
        # Shape (128, 784) because: output_neurons x input_neurons
        # We multiply by 0.01 to keep initial weights small —
        # large initial weights cause exploding gradients
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))   # one bias per hidden neuron

        # W2: connects hidden layer (128) to output layer (10)
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))   # one bias per output neuron

    # ===================== ACTIVATIONS =====================

    def relu(self, z):
        # ReLU: max(0, z) — kills negative values, passes positives through
        # np.maximum works element-wise on the entire array at once
        return np.maximum(0, z)

    def relu_derivative(self, z):
        # Derivative of ReLU: 1 where z > 0, else 0
        # Used during backpropagation
        return (z > 0).astype(float)

    def softmax(self, z):
        # Softmax converts raw scores to probabilities that sum to 1
        # We subtract max(z) for numerical stability — prevents overflow
        # in np.exp() when z contains large numbers. Doesn't change output.
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # ===================== FORWARD PASS =====================

    def forward(self, X):
        """
        X shape: (784, m) where m = number of samples in this batch
        Each column is one flattened image.
        """
        # --- Layer 1: input -> hidden ---
        # Z1 shape: (128, m)
        self.Z1 = self.W1 @ X + self.b1   # weighted sum
        self.A1 = self.relu(self.Z1)       # apply ReLU activation

        # --- Layer 2: hidden -> output ---
        # Z2 shape: (10, m)
        self.Z2 = self.W2 @ self.A1 + self.b2   # weighted sum
        self.A2 = self.softmax(self.Z2)          # apply Softmax — these are probabilities

        # We store Z1, A1, Z2, A2 on self because backprop needs them
        return self.A2

    # ===================== LOSS =====================

    def compute_loss(self, A2, Y):
        """
        Cross-entropy loss.
        A2: predicted probabilities, shape (10, m)
        Y:  true labels as one-hot vectors, shape (10, m)
        """
        m = Y.shape[1]  # number of samples
        # Clip A2 to avoid log(0) which is -infinity
        A2_clipped = np.clip(A2, 1e-8, 1 - 1e-8)
        # Cross-entropy: -sum(Y * log(A2)) averaged over all samples
        loss = -np.sum(Y * np.log(A2_clipped)) / m
        return loss

    # ===================== BACKPROPAGATION =====================

    def backward(self, X, Y):
        """
        Compute gradients of loss with respect to every weight and bias.
        X: input, shape (784, m)
        Y: true labels one-hot, shape (10, m)
        """
        m = X.shape[1]  # number of samples

        # --- Output layer gradient ---
        # For softmax + cross-entropy combined, the gradient simplifies beautifully to:
        # dZ2 = A2 - Y  (predicted probability minus true label)
        dZ2 = self.A2 - Y                          # shape (10, m)
        dW2 = (dZ2 @ self.A1.T) / m               # shape (10, 128)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # shape (10, 1)

        # --- Hidden layer gradient ---
        # Chain rule: error flows back through W2, then through ReLU derivative
        dA1 = self.W2.T @ dZ2                      # shape (128, m)
        dZ1 = dA1 * self.relu_derivative(self.Z1)  # shape (128, m)
        dW1 = (dZ1 @ X.T) / m                     # shape (128, 784)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # shape (128, 1)

        # Store gradients so update() can use them
        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ===================== WEIGHT UPDATE =====================

    def update(self):
        """Gradient descent: nudge every weight in the direction that reduces loss."""
        self.W1 -= self.lr * self.grads["dW1"]
        self.b1 -= self.lr * self.grads["db1"]
        self.W2 -= self.lr * self.grads["dW2"]
        self.b2 -= self.lr * self.grads["db2"]

    # ===================== PREDICT =====================

    def predict(self, X):
        """Return the predicted digit (0-9) for each sample in X."""
        A2 = self.forward(X)
        return np.argmax(A2, axis=0)  # index of highest probability = predicted digit
