# ============================================================
#  CLASSIFICATION TRAINER
# ============================================================

import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ============================================================
# SETTINGS
# ============================================================

CLSROOT = "all_data_cls"
CLASS_FILE = "CLASS_LABELS.txt"

CLS_NUM_CLASSES = 4      # oil, soap, honey, empty
BATCH_SIZE = 4
EPOCHS = 8
LR = 1e-4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

CLASS_NAMES = ["oil", "soap", "honey", "empty"]


def undo_normalisation(img_t):
    """Convert normalised tensor to numpy image."""
    img = img_t.clone()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def stroked(ax, text, fontsize=14):
    """White text with black edge stroke."""
    ax.set_title(
        text,
        fontsize=fontsize,
        color="white",
        pad=6,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )


def show_classification_examples(model, loader, device, max_images=12):
    """
    Display classifier predictions in a 3-column table:

      [IMAGE] | [GT LABEL] | [PRED LABEL]
    """
    model.eval()
    rows = min(max_images, len(loader.dataset))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    shown = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(1)

            imgs_cpu = imgs.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            for i in range(len(imgs)):
                if shown >= rows:
                    plt.tight_layout()
                    plt.show()
                    return

                img_disp = undo_normalisation(imgs_cpu[i])
                gt = CLASS_NAMES[labels_cpu[i]]
                pr = CLASS_NAMES[preds_cpu[i]]

                # --- Column 1: Image ---
                ax_img = axes[shown, 0]
                ax_img.imshow(img_disp)
                stroked(ax_img, f"Image #{shown+1}")
                ax_img.axis("off")

                # --- Column 2: GT Label ---
                ax_gt = axes[shown, 1]
                ax_gt.text(0.5, 0.5, gt,
                           ha="center", va="center",
                           fontsize=18, color="white",
                           path_effects=[pe.withStroke(linewidth=3, foreground="black")])
                ax_gt.set_title("Ground Truth", fontsize=14)
                ax_gt.axis("off")

                # --- Column 3: Predicted Label ---
                ax_pr = axes[shown, 2]
                ax_pr.text(0.5, 0.5, pr,
                           ha="center", va="center",
                           fontsize=18, color="white",
                           path_effects=[pe.withStroke(linewidth=3, foreground="black")])
                ax_pr.set_title("Prediction", fontsize=14)
                ax_pr.axis("off")

                shown += 1

    plt.tight_layout()
    plt.show()
# ============================================================
# SEEDING
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================
# UTILS
# ============================================================

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def load_label_file(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            name, lab = line.strip().split(",")
            mapping[name] = int(lab)
    return mapping

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
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    train = [(f, labels_dict[f]) for f in train_files]
    val   = [(f, labels_dict[f]) for f in val_files]
    test  = [(f, labels_dict[f]) for f in test_files]

    return train, val, test

# ============================================================
# DATASET
# ============================================================

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class ClassificationDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, label = self.items[idx]
        img = Image.open(os.path.join(CLSROOT, name)).convert("RGB")
        img = img_transform(img)
        return img, label

# ============================================================
# MODEL
# ============================================================

def make_classifier():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, CLS_NUM_CLASSES)
    return model

# ============================================================
# TRAINING
# ============================================================

def test_classifier(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return {"acc": correct / total}

def train_classifier(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        model.train()
        correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

        train_acc = correct / total
        val_acc   = test_classifier(model, val_loader, device)["acc"]

        print(f"Epoch {ep+1} | Train {train_acc:.3f} | Val {val_acc:.3f}")

    save_checkpoint(model, "classifier_checkpoint.pt")

# ============================================================
# MAIN
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    labels = load_label_file(CLASS_FILE)
    train_items, val_items, test_items = autosplit(CLSROOT, labels)

    train_ds = ClassificationDataset(train_items)
    val_ds   = ClassificationDataset(val_items)
    test_ds  = ClassificationDataset(test_items)

    model = make_classifier().to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    train_classifier(model, train_loader, val_loader, device)

    print("TEST:", test_classifier(model, test_loader, device))
    # Visualise predictions on test set
    show_classification_examples(model, test_loader, device)

if __name__ == "__main__":
    main()
