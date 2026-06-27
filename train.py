import numpy as np
import os
import urllib.request
import gzip
from model import NeuralNetwork


# ===================== DATA LOADING =====================

def load_mnist():
    """Load MNIST from pre-saved npy files."""
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test  = np.load("data/X_test.npy")
    y_test  = np.load("data/y_test.npy")
    return X_train, y_train, X_test, y_test


def preprocess(X_train, y_train, X_test, y_test):
    """Normalize pixel values and convert labels to one-hot vectors."""
    # Normalize: pixels are 0-255, divide by 255 to get 0.0-1.0
    # This keeps weight updates small and stable
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    # Transpose: we need shape (784, m) not (m, 784)
    # Each column = one image — matches our W @ X convention
    X_train = X_train.T   # (784, 60000)
    X_test  = X_test.T    # (784, 10000)

    # One-hot encode labels: digit 3 -> [0,0,0,1,0,0,0,0,0,0]
    def one_hot(labels, num_classes=10):
        m = labels.shape[0]
        oh = np.zeros((num_classes, m))
        oh[labels, np.arange(m)] = 1
        return oh

    Y_train = one_hot(y_train)   # (10, 60000)
    Y_test  = one_hot(y_test)    # (10, 10000)

    return X_train, Y_train, X_test, Y_test


# ===================== TRAINING LOOP =====================

def train(nn, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=64):
    m = X_train.shape[1]   # total training samples (60000)

    for epoch in range(1, epochs + 1):
        # Shuffle training data at the start of each epoch
        indices = np.random.permutation(m)
        X_shuffled = X_train[:, indices]
        Y_shuffled = Y_train[:, indices]

        epoch_loss = 0
        num_batches = 0

        # Mini-batch gradient descent
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]

            # The full training step in 3 lines:
            A2 = nn.forward(X_batch)
            nn.backward(X_batch, Y_batch)
            nn.update()

            epoch_loss += nn.compute_loss(A2, Y_batch)
            num_batches += 1

        # Evaluate on test set at end of each epoch
        avg_loss = epoch_loss / num_batches
        preds = nn.predict(X_test)
        true_labels = np.argmax(Y_test, axis=0)
        accuracy = np.mean(preds == true_labels) * 100

        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    # Save trained weights
    np.savez("data/weights.npz",
             W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)
    print("\nWeights saved to data/weights.npz")


# ===================== MAIN =====================

if __name__ == "__main__":
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, Y_train, X_test, Y_test = preprocess(X_train, y_train, X_test, y_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    nn = NeuralNetwork(lr=0.1)

    print("\nTraining...\n")
    train(nn, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=64)
