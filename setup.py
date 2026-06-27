import gzip, numpy as np, os

os.makedirs('data', exist_ok=True)

def ri(p):
    with gzip.open(p, 'rb') as f:
        f.read(16)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)

def rl(p):
    with gzip.open(p, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

np.save('data/X_train.npy', ri('data/train-images.gz'))
np.save('data/y_train.npy', rl('data/train-labels.gz'))
np.save('data/X_test.npy',  ri('data/test-images.gz'))
np.save('data/y_test.npy',  rl('data/test-labels.gz'))
print("Done! npy files saved to data/")