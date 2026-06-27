# Handwritten Digit Recognizer

A neural network built **completely from scratch** using only NumPy — no TensorFlow, no PyTorch, no sklearn. Trained on the MNIST dataset to recognize handwritten digits (0–9).

## Architecture

```
Input Layer    →   Hidden Layer     →   Output Layer
784 neurons        128 neurons           10 neurons
(28×28 image)      ReLU activation       Softmax activation
                                         (one per digit 0–9)
```

- **Forward pass:** matrix multiplication + activation functions
- **Loss:** cross-entropy
- **Backpropagation:** chain rule, computed manually
- **Optimizer:** mini-batch gradient descent
- **Achieved accuracy:** ~97% on MNIST test set

## Setup

```bash
pip install numpy pillow
```

## Usage

### Train the network
```bash
python train.py
```
Downloads MNIST automatically, trains for 20 epochs, saves weights to `data/weights.npz`.

### Predict on a custom image
```bash
python predict.py --image path/to/digit.png
```

### Demo on MNIST test samples
```bash
python predict.py --demo
```

## Project Structure

```
mnist_nn/
├── model.py      # NeuralNetwork class (forward pass, backprop, weight update)
├── train.py      # Data loading, preprocessing, training loop
├── predict.py    # CLI for predictions and demo
├── data/         # MNIST files + saved weights (auto-created)
└── README.md
```

## How it works

1. Each 28×28 image is flattened to a vector of 784 pixel values (normalized 0–1)
2. Forward pass multiplies inputs by learned weight matrices through two layers
3. Softmax converts raw scores to probabilities across 10 digit classes
4. Cross-entropy loss measures how wrong the prediction was
5. Backpropagation computes how much each weight contributed to the error
6. Gradient descent nudges every weight to reduce the loss
7. Repeated over 60,000 training examples × 20 epochs
