# ============================================================
#  CLASSIFICATION TRAINER
# ============================================================

import os
import random
from PIL import Image
import numpy as np

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


if __name__ == "__main__":
    main()
