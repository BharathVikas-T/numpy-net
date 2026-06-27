# NumpyNet

A neural network built completely from scratch using only NumPy — no TensorFlow, no PyTorch, no sklearn. Trained on MNIST to recognize handwritten digits (0–9) with **97.8% test accuracy**.

## Architecture

```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

- **Forward pass:** matrix multiplication + activation functions
- **Loss:** cross-entropy
- **Backpropagation:** chain rule, computed manually
- **Optimizer:** mini-batch gradient descent (batch size 64)

## Results

| Epoch | Loss | Test Accuracy |
|-------|------|---------------|
| 1     | 0.54 | 92.0%         |
| 5     | 0.12 | 96.4%         |
| 10    | 0.06 | 97.4%         |
| 20    | 0.03 | **97.8%**     |

## Setup

```bash
pip install numpy
```

Download the 4 MNIST `.gz` files from [yann.lecun.com](http://yann.lecun.com/exdb/mnist/) and place them in a `data/` folder keeping their original names:

```
data/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

## Usage

```bash
python setup.py           # parse MNIST .gz files into numpy arrays
python train.py           # train for 20 epochs (~2 mins)
python predict.py --demo  # run predictions on 10 random test images
python predict.py --image path/to/digit.png  # predict your own image
```

## Project Structure

```
NumpyNet/
├── model.py      # NeuralNetwork class — forward pass, backprop, weight update
├── train.py      # Data loading, preprocessing, training loop
├── predict.py    # CLI for predictions and demo
├── setup.py      # Parses raw MNIST .gz files into numpy arrays
└── README.md
```
