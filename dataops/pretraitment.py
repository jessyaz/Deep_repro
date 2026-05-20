"""
This module is responsible for fetching and pre-processing the MNIST dataset.
It uses the `fetch_openml` function from the `sklearn.datasets` module to download the dataset and store it in a local directory.
As fetch_openml is an experimental feature, it can yield to some network errors.
In that case, you can manually download the dataset from https://www.openml.org/search?type=data&status=active&id=554&sort=runs and put it in the data/ directory
"""

from urllib.request import urlretrieve
import os
import pickle
import numpy as np
import arff


def fetch_mnist_from_openml():
    urlretrieve(
        url="https://www.openml.org/data/download/52667/mnist_784.arff",
        filename="data/mnist_784.arff",
    )


def preprocess_mnist_data():
    with open("data/mnist_784.arff", "r") as f:
        dataset = arff.load(f)
    print(dataset.keys())
    data = np.array(dataset["data"])
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    print(f"X: {X.shape}, y: {y.shape}")
    # Split the data into training and test sets based in the original MNIST split.
    X_train_val, X_test = X[:60000], X[60000:]
    y_train_val, y_test = y[:60000], y[60000:]
    X_train, X_val = X_train_val[:50000], X_train_val[50000:]
    y_train, y_val = y_train_val[:50000], y_train_val[50000:]
    print("Saving MNIST flat dataset")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    # Save the preprocessed data as pickle files.
    mnist_flat = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    with open("data/mnist_flat.pkl", "wb") as f:
        pickle.dump(mnist_flat, f)

    mnist_28_28 = {
        "X_train": X_train.reshape(-1, 28, 28),
        "y_train": y_train,
        "X_val": X_val.reshape(-1, 28, 28),
        "y_val": y_val,
        "X_test": X_test.reshape(-1, 28, 28),
        "y_test": y_test,
    }
    print("MNIST 28x28 dataset")
    print(
        f"X_train: {mnist_28_28['X_train'].shape}, y_train: {mnist_28_28['y_train'].shape}"
    )
    print(f"X_val: {mnist_28_28['X_val'].shape}, y_val: {mnist_28_28['y_val'].shape}")
    print(
        f"X_test: {mnist_28_28['X_test'].shape}, y_test: {mnist_28_28['y_test'].shape}"
    )

    with open("data/mnist_28_28.pkl", "wb") as f:
        pickle.dump(mnist_28_28, f)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/mnist_784.arff"):
        fetch_mnist_from_openml()
    preprocess_mnist_data()
