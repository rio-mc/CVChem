# ---------------------------------------------------------------------
# Classification Trainer
# ---------------------------------------------------------------------
# This script conducts supervised image classification on vial
# contents. All optimisation, validation, and plotting logic is
# delegated to the unified trainer.py module. This ensures parity
# between classification and segmentation experiments and standardises
# experimental reporting.
#
# Workflow overview:
#   - dataset partitioning
#   - model definition (pretrained encoder + custom classifier head)
#   - centralised training loop (from trainer.py)
#   - validation accuracy and confusion matrix (from trainer.py)
#   - qualitative inspection of predictions
# ---------------------------------------------------------------------

import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Unified trainer module
from trainer import (
    train_loop,
    plot_training_curves,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------
# Experiment Parameters
# ---------------------------------------------------------------------

CLSROOT = "all_data_cls"
CLASS_FILE = "CLASS_LABELS.txt"

CLS_NUM_CLASSES = 4
BATCH_SIZE = 4
EPOCHS = 8
LR = 1e-4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

CLASS_NAMES = ["oil", "soap", "honey", "empty"]


# ---------------------------------------------------------------------
# Reproducibility Controls
# ---------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ---------------------------------------------------------------------
# Normalisation Reversal for Visualisation
# ---------------------------------------------------------------------

def undo_normalisation(img_t):
    img = img_t.clone()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------
# Overlay Styling
# ---------------------------------------------------------------------

def stroked(ax, text):
    ax.set_title(
        text,
        fontsize=14,
        color="white",
        pad=6,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )


# ---------------------------------------------------------------------
# Visual Demonstration of Classifier Behaviour
# ---------------------------------------------------------------------

def show_classification_examples(model, loader, device, max_images=24, cols=4):
    """
    Display input images paired with ground truth and predicted labels.
    Designed for laboratory-style qualitative review.
    """

    model.eval()

    images, gts, preds = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pr = model(imgs).argmax(1)

            for i in range(len(imgs)):
                if len(images) >= max_images:
                    break
                images.append(undo_normalisation(imgs[i].cpu()))
                gts.append(CLASS_NAMES[labels[i].item()])
                preds.append(CLASS_NAMES[pr[i].item()])

            if len(images) >= max_images:
                break

    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]

        if i >= n:
            ax.axis("off")
            continue

        img = images[i]
        gt = gts[i]
        pr = preds[i]

        ax.imshow(img)
        ax.axis("off")

        # Brightness-based placement of text
        brightness = img[:50, :, :].mean()
        ypos = 0.05 if brightness < 0.55 else 0.20

        ax.text(
            0.02, ypos,
            f"GT: {gt} | Pred: {pr}",
            color="white",
            fontsize=11,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            weight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")]
        )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def load_label_file(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            name, lab = line.strip().split(",")
            mapping[name] = int(lab)
    return mapping


# ---------------------------------------------------------------------
# Dataset Partitioning
# ---------------------------------------------------------------------

def autosplit(root, labels_dict):
    files = sorted([
        f for f in os.listdir(root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    random.shuffle(files)
    n = len(files)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    train = [(f, labels_dict[f]) for f in train_files]
    val   = [(f, labels_dict[f]) for f in val_files]
    test  = [(f, labels_dict[f]) for f in test_files]

    return train, val, test


# ---------------------------------------------------------------------
# Transform Pipeline
# ---------------------------------------------------------------------

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------

class ClassificationDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        name, label = self.items[idx]
        path = os.path.join(CLSROOT, name)
        img = Image.open(path).convert("RGB")
        img = img_transform(img)
        return img, label


# ---------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------

def make_classifier():
    """
    ResNet-18 backbone with a custom 4-class output layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, CLS_NUM_CLASSES)
    return model


# ---------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    labels = load_label_file(CLASS_FILE)
    train_items, val_items, test_items = autosplit(CLSROOT, labels)

    train_ds = ClassificationDataset(train_items)
    val_ds   = ClassificationDataset(val_items)
    test_ds  = ClassificationDataset(test_items)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE)
    test_loader  = DataLoader(test_ds, BATCH_SIZE)

    model = make_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------------------------------------------
    # Unified training loop (from trainer.py)
    # -------------------------------------------------------------
    log = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimiser=optimiser,
        epochs=EPOCHS,
        task_type="classification",
        num_classes=CLS_NUM_CLASSES
    )

    # Training curves
    plot_training_curves(log, metric_name="Validation Accuracy")

    # Confusion matrix (final epoch)
    if log.conf_mats:
        plot_confusion_matrix(log.conf_mats[-1], CLASS_NAMES)

    # Qualitative inspection
    show_classification_examples(model, test_loader, device)

    save_checkpoint(model, "classifier_checkpoint.pt")


if __name__ == "__main__":
    main()
