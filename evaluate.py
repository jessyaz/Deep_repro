"""
Evaluate a trained model from an MLflow run_id.
Recovers the best checkpoint and the config from the run, then evaluates the model on the test set
and logs a metrics report back as an artifact to the same run.
"""

import tempfile
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.artifacts import download_artifacts
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn.metrics import classification_report, confusion_matrix
from dataops.dataobject import DataObject


##TODO 7##
# complete the parameters of the hydra decorator following the project structure
@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    ##TODO 11##
    # Once everything is working consistently locally,
    # Uncomment this line to set the MLflow tracking to the global performance
    # dashboard and compete with your peers !
    # mlflow.set_tracking_uri("http://172.19.1.76:4201")

    with tempfile.TemporaryDirectory() as tmp:
        # Recover config and best checkpoint from the run

        ##TODO 8##
        # Complete the artifacts_uri for the config file and the best set of parameters
        exp_cfg = OmegaConf.load(
            download_artifacts(
                artifact_uri=f"runs:/{cfg.run_id}/<complete here>",
                dst_path=tmp,
            )
        )
        ckpt_path = download_artifacts(
            artifact_uri=f"runs:/{cfg.run_id}/checkpoints/<complete here>",
            dst_path=tmp,
        )

        ##TODO 9##
        # Instantiate the data object from the hydra config
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
        report_dict = cast(
            dict, classification_report(all_labels, all_preds, output_dict=True)
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
                cast(str, classification_report(all_labels, all_preds)),
                "classification_report.txt",
            )

            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, ax=ax)
            classes = np.arange(cm.shape[0])
            ax.set(
                xticks=classes,
                yticks=classes,
                xlabel="Predicted label",
                ylabel="True label",
                title="Confusion Matrix",
            )

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

            fig.tight_layout()

            ##TODO 10##
            # log the confusion matrix as an artifact to MLflow

            plt.close(fig)


if __name__ == "__main__":
    main()
