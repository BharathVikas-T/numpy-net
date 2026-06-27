import numpy as np
import argparse
import sys
import os


def load_model():
    """Load saved weights into a NeuralNetwork instance."""
    from model import NeuralNetwork
    if not os.path.exists("data/weights.npz"):
        print("Error: No trained weights found. Run train.py first.")
        sys.exit(1)
    weights = np.load("data/weights.npz")
    nn = NeuralNetwork()
    nn.W1, nn.b1 = weights["W1"], weights["b1"]
    nn.W2, nn.b2 = weights["W2"], weights["b2"]
    return nn


def predict_image(image_path):
    """Load a PNG/JPG image, preprocess it, and predict the digit."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed. Run: pip install pillow")
        sys.exit(1)

    img = Image.open(image_path).convert("L")   # convert to grayscale
    img = img.resize((28, 28))                   # resize to 28x28
    pixels = np.array(img, dtype=np.float64)

    # MNIST has white digits on black background.
    # If your image is black on white, invert it:
    if pixels.mean() > 127:
        pixels = 255 - pixels

    pixels = pixels / 255.0          # normalize to [0, 1]
    X = pixels.reshape(784, 1)       # shape (784, 1) — one image, one column

    nn = load_model()
    probs = nn.forward(X)            # shape (10, 1)
    predicted_digit = int(np.argmax(probs, axis=0)[0])
    confidence = float(probs[predicted_digit, 0]) * 100

    print(f"Predicted digit : {predicted_digit}")
    print(f"Confidence      : {confidence:.1f}%")
    print("\nAll probabilities:")
    for digit, prob in enumerate(probs[:, 0]):
        bar = "█" * int(prob * 30)
        marker = " <-- predicted" if digit == predicted_digit else ""
        print(f"  {digit}: {bar:<30} {prob*100:.1f}%{marker}")

    return predicted_digit


def visualize_mnist_predictions():
    """Show predictions on 10 random MNIST test images (no external image needed)."""
    if not os.path.exists("data/X_test.npy"):
        print("MNIST test data not found. Run train.py first.")
        sys.exit(1)

    X_test = np.load("data/X_test.npy") / 255.0
    y_test = np.load("data/y_test.npy")

    nn = load_model()

    # Pick 10 random test images
    indices = np.random.choice(len(y_test), 10, replace=False)
    print(f"{'Index':<8} {'True':>6} {'Predicted':>10} {'Correct':>8}")
    print("-" * 36)
    correct = 0
    for idx in indices:
        x = X_test[idx].reshape(784, 1)
        pred = int(np.argmax(nn.forward(x), axis=0)[0])
        true = int(y_test[idx])
        ok = "✓" if pred == true else "✗"
        if pred == true:
            correct += 1
        print(f"{idx:<8} {true:>6} {pred:>10} {ok:>8}")
    print(f"\nAccuracy on this sample: {correct}/10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognizer")
    parser.add_argument("--image", type=str, help="Path to an image file (PNG/JPG)")
    parser.add_argument("--demo",  action="store_true",
                        help="Run predictions on 10 random MNIST test images")
    args = parser.parse_args()

    if args.image:
        predict_image(args.image)
    elif args.demo:
        visualize_mnist_predictions()
    else:
        parser.print_help()
