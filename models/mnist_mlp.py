import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_layers,
                 output_dim,
                 dropout):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim),
                  nn.ReLU(),
                  nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out = self.mlp(X)
        y_hat = self.linear(out)
        return y_hat