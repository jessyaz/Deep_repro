"""
TODO: @J-ally I propose U a basic architecture, feel free of change it to fill our performence and computing needs.
"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_dim,
                 n_layers,
                 output_dim,
                 dropout):
        super().__init__()

    def forward(self, X):
        out = self.cnn(X)
        out = out.view(out.size(0), -1)
        y_hat = self.linear(out)
        return y_hat