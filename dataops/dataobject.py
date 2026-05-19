import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 flattened: bool = False):
        self.np_X = X
        self.np_y = y
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.flattened = flattened

    def plot_example(self):
        """
        Plot the first 5 images and their labels in a row.
        """
        if self.flattened:
            X = self.np_X.reshape(-1, 28, 28)
        else:
            X = self.np_X
        for i, (img, y) in enumerate(zip(X[:5], self.np_y[:5])):
            plt.subplot(151 + i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class DataObject:
    def __init__(self,
                 data_path: str,
                 flattened: bool = False,
                 batch_size: int = 128):
        self.data_path = data_path
        self.flattened = flattened
        self.batch_size = batch_size

    def _normalize(self, X_train, X_val, X_test):
        return X_train / 255.0, X_val / 255.0, X_test / 255.0
    
    def make_datasets(self) -> None:
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        X_train, X_val, X_test = self._normalize(X_train, X_val, X_test)

        self.train_ds = MNISTDataset(X_train,
                                     y_train,
                                     flattened=self.flattened)
        self.val_ds = MNISTDataset(X_val,
                                   y_val,
                                   flattened=self.flattened)
        self.test_ds = MNISTDataset(X_test,
                                    y_test,
                                    flattened=self.flattened)


    def make_dataloaders(self) -> None:
        self.train_dl = DataLoader(self.train_ds,
                                   batch_size=self.batch_size,
                                   shuffle=True)
        self.val_dl = DataLoader(self.val_ds,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        self.test_dl = DataLoader(self.test_ds,
                                  batch_size=self.batch_size,
                                  shuffle=False)