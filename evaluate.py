"""
Evaluate a trained model from an MLflow run_id.
Recovers checkpoint and config from the run, evaluates on the test set,
and logs a metrics report back as an artifact to the same run.
"""

import tempfile
import numpy as np
import mlflow
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataops.dataobject import DataObject


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # mlflow.set_tracking_uri("https://mlflow-happyr.allynd.re/")

    with tempfile.TemporaryDirectory() as tmp:
        # Recover config and best checkpoint from the run
        exp_cfg = OmegaConf.load(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{cfg.run_id}/config_exp.yaml",
                dst_path=tmp,
            )
        )
        ckpt_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{cfg.run_id}/checkpoints/best_checkpoint.pt",
            dst_path=tmp,
        )

        # Rebuild data pipeline from the recovered config file
        data_object = DataObject(
            data_path=exp_cfg.data.data_path,
            flattened=exp_cfg.data.flattened,
            batch_size=exp_cfg.training.batch_size,
        )
        data_object.make_datasets()
        data_object.make_dataloaders()

        # Rebuild model and load weights
        model = instantiate(exp_cfg.model.model)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Inference on test set
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in data_object.test_dl:
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                all_labels.extend(y_batch.numpy())
                all_preds.extend(torch.argmax(probs, dim=1).numpy())
                all_probs.extend(probs.numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Compute metrics
        report_dict = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )

        print(f"Accuracy: {report_dict['accuracy']:.4f}")
        print(f"Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report_dict['weighted avg']['f1-score']:.4f}")

        # Push report to the same MLflow run
        with mlflow.start_run(run_id=cfg.run_id):
            mlflow.log_metrics(
                {
                    "test/accuracy": float(report_dict["accuracy"]),
                    "test/macro_f1": float(report_dict["macro avg"]["f1-score"]),
                    "test/weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
                    "test/macro_precision": float(
                        report_dict["macro avg"]["precision"]
                    ),
                    "test/macro_recall": float(report_dict["macro avg"]["recall"]),
                }
            )
            mlflow.log_text(
                classification_report(all_labels, all_preds, zero_division=0),
                "classification_report.txt",
            )


if __name__ == "__main__":
    main()
