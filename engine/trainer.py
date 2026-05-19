import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from dataops.dataobject import DataObject

def train(model,
          data_object: DataObject,
          epochs=10,
          optimizer: optim.Optimizer = None,
          criterion: nn.Module = None,
          checkpoint_dir: str = None):

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

        mlflow.log_metrics({"train/loss": avg_loss,
                            "val/accuracy": accuracy},
                            step=epoch)

        if checkpoint_dir is not None:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

            if (epoch + 1) % 2 == 0:
                periodic_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), periodic_path)
                mlflow.log_artifact(periodic_path, artifact_path="checkpoints")
