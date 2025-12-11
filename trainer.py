# ---------------------------------------------------------------
# trainer.py
# Unified training, evaluation, and plotting utilities for both
# classification and segmentation models.
#
# Provides:
#   - TrainingLog (stores epoch-wise loss + metric + confusion matrix)
#   - evaluate_classifier()
#   - evaluate_segmentation()
#   - train_loop()  — task-agnostic training engine
#   - plot_training_curves()
#   - plot_confusion_matrix()
#
# This script is intentionally modality-neutral to support both
# classification and segmentation tasks with minimal redundancy.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------
# Training log structure
# ---------------------------------------------------------------

class TrainingLog:
    """
    Structured record of training dynamics:
        - train_loss: list of mean training losses per epoch
        - val_metric: list of validation metrics (accuracy or pixel accuracy)
        - conf_mats:  list of confusion matrices (classification only)
    """

    def __init__(self):
        self.train_loss = []
        self.val_metric = []
        self.conf_mats  = []

    def record(self, train_loss, val_metric, conf_mat=None):
        self.train_loss.append(train_loss)
        self.val_metric.append(val_metric)
        if conf_mat is not None:
            self.conf_mats.append(conf_mat)


# ---------------------------------------------------------------
# Classification evaluation — accuracy + confusion matrix
# ---------------------------------------------------------------

def evaluate_classifier(model, loader, device, num_classes: int):
    model.eval()
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            total   += len(labels)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = correct / total
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return acc, conf_mat


# ---------------------------------------------------------------
# Segmentation evaluation — pixel accuracy
# ---------------------------------------------------------------

def evaluate_segmentation(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            preds = logits.argmax(1)

            correct += (preds == masks).sum().item()
            total   += masks.numel()

    return correct / total


# ---------------------------------------------------------------
# Generic training engine
# ---------------------------------------------------------------

def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimiser,
    epochs: int,
    task_type: str,
    num_classes: int = None
):
    """
    Unified training procedure for both tasks.

    Parameters:
        task_type: "classification" or "segmentation"
        num_classes: required only for classification

    Returns:
        TrainingLog instance
    """

    log = TrainingLog()

    for ep in range(epochs):
        model.train()
        epoch_losses = []

        # -------------------------
        #  Training pass
        # -------------------------
        for batch in train_loader:
            optimiser.zero_grad()

            if task_type == "classification":
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

            elif task_type == "segmentation":
                imgs, masks = batch
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                loss = criterion(logits, masks)

            else:
                raise ValueError(f"Unknown task_type: {task_type}")

            loss.backward()
            optimiser.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))

        # -------------------------
        #  Validation pass
        # -------------------------
        if task_type == "classification":
            val_metric, conf_mat = evaluate_classifier(
                model, val_loader, device, num_classes
            )
            log.record(mean_loss, val_metric, conf_mat)

            print(
                f"Epoch {ep+1}/{epochs} | "
                f"Loss {mean_loss:.4f} | "
                f"Val Accuracy {val_metric:.4f}"
            )

        else:  # segmentation
            val_metric = evaluate_segmentation(model, val_loader, device)
            log.record(mean_loss, val_metric)

            print(
                f"Epoch {ep+1}/{epochs} | "
                f"Loss {mean_loss:.4f} | "
                f"Val Pixel Acc {val_metric:.4f}"
            )

    return log


# ---------------------------------------------------------------
# Plot: training curves
# ---------------------------------------------------------------

def plot_training_curves(log: TrainingLog, metric_name="Validation Metric"):
    epochs = np.arange(1, len(log.train_loss) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # -----------------------------
    # Left panel: Training loss
    # -----------------------------
    ax[0].plot(epochs, log.train_loss, marker="o")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[0].minorticks_on()

    ax[0].grid(which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    ax[0].grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.20)

    # -----------------------------
    # Right panel: Validation metric
    # -----------------------------
    ax[1].plot(epochs, log.val_metric, marker="o")
    ax[1].set_title(metric_name)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(metric_name)

    ax[1].minorticks_on()

    ax[1].grid(which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    ax[1].grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.20)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Plot: confusion matrix (classification)
# ---------------------------------------------------------------

def plot_confusion_matrix(conf_mat, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
