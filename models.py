import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
#                   USER SETTINGS
# ============================================================

TASK = "cls"   # "cls" or "seg"

ROOT = "all_data"      # folder containing all images (+ masks if segmentation)
CLASS_FILE = "CLASS_LABELS.txt" # master label file for classification
SEG_FILE = "SEG_LABELS.txt" # master label file for segmentation

BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


# ============================================================
#                        SEEDING
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
#                   CHECKPOINT UTILITIES
# ============================================================

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Checkpoint loaded from {path}")
    else:
        print("No checkpoint found – starting fresh.")
    return model


# ============================================================
#                   DATA WRANGLER
# ============================================================

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)
])


def load_label_file(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            fname, lab = line.strip().split(",")
            mapping[fname] = int(lab)
    return mapping


def autosplit(root, labels_dict, task):
    files = sorted([
        f for f in os.listdir(root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if task == "seg":
        files = [f for f in files if not f.endswith("_mask.png")]

    random.shuffle(files)
    N = len(files)

    if N < 3:
        raise ValueError("Need at least 3 images for train/val/test.")

    n_train = int(N * TRAIN_RATIO)
    n_val   = int(N * VAL_RATIO)
    n_test  = N - n_train - n_val

    if n_train == 0: n_train = 1
    if n_val == 0: n_val = 1
    if n_test == 0: n_test = 1

    while n_train + n_val + n_test > N:
        n_train -= 1
        if n_train == 0:
            break

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    if task == "cls":
        train = [(f, labels_dict[f]) for f in train_files]
        val   = [(f, labels_dict[f]) for f in val_files]
        test  = [(f, labels_dict[f]) for f in test_files]
    else:
        train, val, test = train_files, val_files, test_files

    return train, val, test


class LiquidClassificationDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, label = self.items[idx]
        img = Image.open(os.path.join(ROOT, name)).convert("RGB")
        img = img_transform(img)
        return img, label


class LiquidSegmentationDataset(Dataset):
    def __init__(self, filenames):
        self.names = filenames

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(ROOT, name)
        mask_path = img_path.replace(".png", "_mask.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = img_transform(img)
        mask = mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


# ============================================================
#                   MODELS
# ============================================================

def make_classifier(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.u3 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u1 = DoubleConv(128 + 64, 64)

        self.final = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.d1(x);    p1 = self.pool(c1)
        c2 = self.d2(p1);   p2 = self.pool(c2)
        c3 = self.d3(p2);   p3 = self.pool(c3)
        c4 = self.d4(p3)

        up3 = self.u3(torch.cat([F.interpolate(c4, scale_factor=2, mode="bilinear", align_corners=False), c3], dim=1))
        up2 = self.u2(torch.cat([F.interpolate(up3, scale_factor=2, mode="bilinear", align_corners=False), c2], dim=1))
        up1 = self.u1(torch.cat([F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False), c1], dim=1))

        return self.final(up1)


# ============================================================
#                 TRAINING + TESTING
# ============================================================

def train_classifier(model, loader_train, loader_val, device):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    ckpt_path = "classifier_checkpoint.pt"

    history = {
        "epoch": [],
        "train_acc": [],
        "val_acc": []
    }

    for ep in range(EPOCHS):
        model.train()
        total, correct = 0, 0

        for imgs, labels in loader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optim.step()

            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        val_acc = test_classifier(model, loader_val, device)["acc"]

        print(f"Epoch {ep+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        # Save history
        history["epoch"].append(ep + 1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save checkpoint
        save_checkpoint(model, ckpt_path)

    # Write CSV log
    import csv
    with open("training_log_cls.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "val_acc"])
        for e, ta, va in zip(history["epoch"], history["train_acc"], history["val_acc"]):
            writer.writerow([e, ta, va])

    print("Training log written to training_log_cls.csv")

    return history



def test_classifier(model, loader, device):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return {"acc": correct / total}

def train_seg(model, loader_train, loader_val, device):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    ckpt_path = "unet_checkpoint.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_pixel_acc": []
    }

    for ep in range(EPOCHS):
        model.train()
        losses = []

        for imgs, masks in loader_train:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optim.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optim.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_pixel_acc = test_seg(model, loader_val, device)["pixel_acc"]

        print(f"Epoch {ep+1} | Train loss: {train_loss:.4f} | Val pixel acc: {val_pixel_acc:.4f}")

        history["epoch"].append(ep + 1)
        history["train_loss"].append(train_loss)
        history["val_pixel_acc"].append(val_pixel_acc)

        save_checkpoint(model, ckpt_path)

    # Write CSV log
    import csv
    with open("training_log_seg.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_pixel_acc"])
        for e, tl, vp in zip(history["epoch"], history["train_loss"], history["val_pixel_acc"]):
            writer.writerow([e, tl, vp])

    print("Training log written to training_log_seg.csv")

    return history

def test_seg(model, loader, device):
    model.eval()
    total_pix, correct_pix = 0, 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs).argmax(1)
            correct_pix += (preds == masks).sum().item()
            total_pix += masks.numel()

    return {"pixel_acc": correct_pix / total_pix}

def confusion_matrix_cls(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs).argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)
    return cm

import matplotlib.pyplot as plt

def show_test_predictions(model, test_loader, device, class_names=None, max_cols=4):
    model.eval()

    imgs_list = []
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs).argmax(1)

            imgs_list.extend(imgs.cpu())
            preds_list.extend(preds.cpu())
            labels_list.extend(labels.cpu())

    N = len(imgs_list)
    cols = max_cols
    rows = (N + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(N):
        img = imgs_list[i]
        pred = preds_list[i].item()
        label = labels_list[i].item()

        # Undo normalisation for display
        img_disp = img.clone()
        img_disp = img_disp * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_disp = img_disp + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        img_disp = img_disp.clamp(0, 1)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img_disp.permute(1, 2, 0))
        ax.axis("off")

        if class_names:
            title = f"P: {class_names[pred]} | T: {class_names[label]}"
        else:
            title = f"P: {pred} | T: {label}"

        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.show()


# ============================================================
#                        MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if TASK == "seg":
        labels_dict = load_label_file(SEG_FILE)
    else:
        labels_dict = load_label_file(CLASS_FILE)

    train_items, val_items, test_items = autosplit(ROOT, labels_dict, TASK)
    print("TRAIN:", len(train_items))
    print("VAL:", len(val_items))
    print("TEST:", len(test_items))

    # Create datasets
    if TASK == "cls":
        train_ds = LiquidClassificationDataset(train_items)
        val_ds   = LiquidClassificationDataset(val_items)
        test_ds  = LiquidClassificationDataset(test_items)

        model = make_classifier(NUM_CLASSES).to(device)

    else:
        train_ds = LiquidSegmentationDataset(train_items)
        val_ds   = LiquidSegmentationDataset(val_items)
        test_ds  = LiquidSegmentationDataset(test_items)

        model = UNet(NUM_CLASSES).to(device)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Train
    if TASK == "cls":
        train_classifier(model, train_loader, val_loader, device)
        print("TEST:", test_classifier(model, test_loader, device))
        confusion_matrix_cls(model, test_loader, device)

        class_names = ["oil", "soap", "honey", "empty"]
        show_test_predictions(model, test_loader, device, class_names)

    else:
        train_seg(model, train_loader, val_loader, device)
        print("TEST:", test_seg(model, test_loader, device))

if __name__ == "__main__":
    main()
