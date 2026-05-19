"""
This script launch a training experiment using the config files.
"""
import os
import mlflow
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from engine.trainer import train
from dataops.dataobject import DataObject

@hydra.main(config_path='conf',
            config_name='config',
            version_base='1.3')
def main(cfg: DictConfig):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    os.makedirs(cfg.checkpoints_dir + cfg.exp_name,
                exist_ok=True)

    data_object = DataObject(data_path=cfg.data.data_path,
                             flattened=cfg.data.flattened,
                             batch_size=cfg.training.batch_size)
    data_object.make_datasets()
    data_object.make_dataloaders()

    model = instantiate(cfg.model.model)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run():
        mlflow.set_tag("exp_name", cfg.exp_name)
        # Log config as an artifact
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_exp.yaml")
        # Log training parameters
        for key, value in cfg.training.items():
            mlflow.log_param(key, value)
        # Log model hyperparameters
        for key, value in cfg.model.hyperparameters.items():
            mlflow.log_param(key, value)

        train(model,
              data_object=data_object,
              epochs=cfg.training.epochs,
              optimizer=optimizer,
              criterion=criterion,
              checkpoint_dir=cfg.checkpoints_dir + cfg.exp_name)

if __name__ == "__main__":
    main()