"""
This script launch a training experiment using the config files.
"""

import os
import mlflow
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dataops.dataobject import DataObject
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


###############################################################################
#                                  MODELS                                     #
###############################################################################


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out = self.mlp(X)
        y_hat = self.linear(out)
        return y_hat


# For running a train with the CNN model: uv run train.py model=cnn data=data_2d exp_name=cnn_experiment
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, hidden_dim, n_layers, output_dim, dropout):
        super().__init__()
        # First conv block: input_channels -> hidden_dim
        layers = [
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        ]
        # Additional conv blocks: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout),
            ]
        self.cnn = nn.Sequential(*layers)
        # After n_layers MaxPool2d(2) on a 28x28 input: spatial size = 28 // (2 ** n_layers)
        spatial = 28 // (2**n_layers)
        self.linear = nn.Linear(hidden_dim * spatial * spatial, output_dim)

    def forward(self, X):
        out = self.cnn(X)
        out = out.view(out.size(0), -1)
        y_hat = self.linear(out)
        return y_hat


###############################################################################
#                                  TRAINER                                    #
###############################################################################


def train(
    model,
    data_object: DataObject,
    epochs=10,
    optimizer: optim.Optimizer = None,
    criterion: nn.Module = None,
    checkpoint_dir: str = None,
) -> float:
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in data_object.train_dl:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(data_object.train_dl)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in data_object.val_dl:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            accuracy = correct / total

        mlflow.log_metrics(
            {"train/loss": avg_loss, "val/accuracy": accuracy}, step=epoch
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(model.state_dict(), best_path)
            ##TODO 3##
            # log the best checkpoint as an artifact into the "checkpoints" artifact folder to MLflow

        if (epoch + 1) % 2 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), periodic_path)
            ##TODO 4##
            # log the periodic checkpoint as an artifact into the "checkpoints" artifact folder to MLflow

    return best_accuracy


###############################################################################
#                               MAIN FUNCTION                                 #
###############################################################################


##TODO 2##
# complete the parameters of the hydra decorator following the project structure
@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(cfg.checkpoints_dir + cfg.exp_name, exist_ok=True)

    ##TODO 5##
    # Instantiate the data object from the hydra config
    data_object.make_datasets()
    data_object.make_dataloaders()

    model = instantiate(cfg.model.model)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion)

    ##TODO 11##
    # Once everything is working consistently locally,
    # Uncomment this line to set the MLflow tracking to the global performance
    # dashboard and compete with your peers !
    # mlflow.set_tracking_uri("http://172.19.1.76:4201")

    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run():
        mlflow.set_tag("exp_name", cfg.exp_name)
        # Log config as an artifact
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_exp.yaml")

        ##TODO 6##
        # Log the training parameters and model hyperparameters to MLflow

        best_val_accuracy = train(
            model,
            data_object=data_object,
            epochs=cfg.training.epochs,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=cfg.checkpoints_dir + cfg.exp_name,
        )
        mlflow.log_metric("val/best_accuracy", best_val_accuracy)

    return best_val_accuracy


if __name__ == "__main__":
    main()
