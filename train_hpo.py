"""
HPO entry point using Hydra + Optuna sweeper.

Run:
    uv run train_hpo.py --config-name mlp_hpo --multirun
    uv run train_hpo.py --config-name cnn_hpo --multirun
"""

import os
import mlflow
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dataops.dataobject import DataObject

from train import train


@hydra.main(config_path="conf", config_name="mlp_hpo", version_base="1.3")
def main(cfg: DictConfig) -> float:
    torch.manual_seed(42)

    # Pair data to model: CNNs need 2-D tensors, MLPs need flat vectors
    is_cnn = "train.SimpleCNN" == cfg.model.model._target_
    data_path = "data/fmnist_28_28.pkl" if is_cnn else "data/fmnist_flat.pkl"

    data_object = DataObject(
        data_path=data_path,
        flattened=not is_cnn,
        batch_size=cfg.training.batch_size,
    )
    data_object.make_datasets()
    data_object.make_dataloaders()

    model = instantiate(cfg.model.model)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion)

    # mlflow.set_tracking_uri("https://mlflow-happyr.allynd.re/")
    mlflow.set_experiment(cfg.exp_name)

    with mlflow.start_run():
        mlflow.set_tag("model_type", cfg.model.model._target_)
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_hpo.yaml")
        for k, v in cfg.training.items():
            mlflow.log_param(k, v)
        for k, v in cfg.model.hyperparameters.items():
            mlflow.log_param(k, v)

        checkpoint_dir = cfg.checkpoints_dir + cfg.exp_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_accuracy = train(
            model,
            data_object=data_object,
            epochs=cfg.training.epochs,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
        )
        mlflow.log_metric("val/best_accuracy", best_val_accuracy)

    # Returned value is maximized by Optuna (direction=maximize in the hpo config)
    return best_val_accuracy


if __name__ == "__main__":
    main()
