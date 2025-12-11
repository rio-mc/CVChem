# ============================================================
#  SEGMENTATION TRAINER
# ============================================================

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ============================================================
# SETTINGS
# ============================================================

SEGROOT = "all_data_seg"
SEG_FILE = "SEG_LABELS.txt"     # filenames only

# Pixel-level classes:
# 0 background
# 1 honey
# 2 oil
# 3 vial interior
# 4 soap
SEG_NUM_CLASSES = 5

BATCH_SIZE = 4
EPOCHS = 8
LR = 1e-4

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15

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
    print(f"Saved checkpoint to {path}")

def load_label_file(path):
    # Only listing filenames, not used for semantic labels
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            name, _ = line.strip().split(",")
            mapping[name] = 0
    return mapping

def autosplit(root, labels_dict):
    files = sorted([
        f for f in os.listdir(root)
        if f.lower().endswith((".png",".jpg",".jpeg"))
           and "_mask" not in f
    ])
    random.shuffle(files)
    n = len(files)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    return (
        files[:n_train],
        files[n_train:n_train+n_val],
        files[n_train+n_val:]
    )

# ============================================================
# TRANSFORMS + COLOUR MAP
# ============================================================

img_transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

mask_transform = T.Compose([
    T.Resize((256,256), interpolation=T.InterpolationMode.NEAREST)
])

CLASS_COLOURS = {
    0: (0,0,0),         # background
    1: (0,0,255),       # honey
    2: (255,0,0),       # oil
    3: (180,180,180),   # interior
    4: (0,255,0),       # soap (future)
}

def mask_to_colour(mask):
    mask = np.asarray(mask)
    h,w = mask.shape
    col = np.zeros((h,w,3), dtype=np.uint8)
    for cid, bgr in CLASS_COLOURS.items():
        col[mask == cid] = bgr
    return col

def draw_title(ax, text):
    ax.set_title(text, fontsize=14, color="white",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

# ============================================================
# DATASET
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path  = os.path.join(SEGROOT, name)
        mask_path = os.path.splitext(img_path)[0] + "_mask.png"

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img  = img_transform(img)
        mask = torch.from_numpy(np.array(mask_transform(mask))).long()

        return img, mask

# ============================================================
# MODEL (UNet)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1), nn.ReLU(True),
            nn.Conv2d(out_c,out_c,3,padding=1), nn.ReLU(True)
        )
    def forward(self,x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.down1 = DoubleConv(3,64)
        self.down2 = DoubleConv(64,128)
        self.down3 = DoubleConv(128,256)
        self.down4 = DoubleConv(256,512)

        self.pool = nn.MaxPool2d(2)

        self.up3 = DoubleConv(256+512,256)
        self.up2 = DoubleConv(128+256,128)
        self.up1 = DoubleConv(64+128,64)

        self.final = nn.Conv2d(64,num_classes,1)

    def forward(self,x):
        c1 = self.down1(x); p1 = self.pool(c1)
        c2 = self.down2(p1); p2 = self.pool(c2)
        c3 = self.down3(p2); p3 = self.pool(c3)
        c4 = self.down4(p3)

        u3 = self.up3(torch.cat([F.interpolate(c4,scale_factor=2,mode="bilinear",align_corners=False), c3], dim=1))
        u2 = self.up2(torch.cat([F.interpolate(u3,scale_factor=2,mode="bilinear",align_corners=False), c2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2,scale_factor=2,mode="bilinear",align_corners=False), c1], dim=1))
        return self.final(u1)

# ============================================================
# TRAINING
# ============================================================

def test_seg(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == masks).sum().item()
            total   += masks.numel()
    return {"pixel_acc": correct / total}

def train_seg(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        model.train()
        losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_acc    = test_seg(model, val_loader, device)["pixel_acc"]
        print(f"Epoch {ep+1} | Loss {train_loss:.3f} | Val Pixel Acc {val_acc:.3f}")

    save_checkpoint(model, "unet_checkpoint.pt")

# ============================================================
# VISUALISATION
# ============================================================

def show_segmentation_examples(model, loader, device, max_images=6):
    model.eval()
    shown = 0
    fig = plt.figure(figsize=(14,5*max_images))

    for imgs, masks in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs).argmax(1).cpu().numpy()

        imgs_cpu  = imgs.cpu()
        masks_cpu = masks.numpy()

        for i in range(len(imgs)):
            if shown >= max_images:
                plt.tight_layout(); plt.show(); return

            r = shown * 3
            # Undo normalisation
            img = imgs_cpu[i]
            img_disp = img * torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            img_disp += torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            img_disp = img_disp.clamp(0,1).permute(1,2,0).numpy()

            true_mask = masks_cpu[i]
            pred_mask = preds[i]

            ax1 = plt.subplot(max_images,3,r+1)
            ax1.imshow(img_disp); draw_title(ax1,"Original"); ax1.axis("off")

            ax2 = plt.subplot(max_images,3,r+2)
            ax2.imshow(mask_to_colour(true_mask)); draw_title(ax2,"Ground Truth"); ax2.axis("off")

            ax3 = plt.subplot(max_images,3,r+3)
            ax3.imshow(mask_to_colour(pred_mask)); draw_title(ax3,"Prediction"); ax3.axis("off")

            shown += 1

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    labels = load_label_file(SEG_FILE)
    train_files, val_files, test_files = autosplit(SEGROOT, labels)

    print("TRAIN:", len(train_files))
    print("VAL:",   len(val_files))
    print("TEST:",  len(test_files))

    train_ds = SegmentationDataset(train_files)
    val_ds   = SegmentationDataset(val_files)
    test_ds  = SegmentationDataset(test_files)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE)
    test_loader  = DataLoader(test_ds, BATCH_SIZE)

    model = UNet(SEG_NUM_CLASSES).to(device)

    train_seg(model, train_loader, val_loader, device)

    print("TEST:", test_seg(model, test_loader, device))

    show_segmentation_examples(model, test_loader, device)


if __name__ == "__main__":
    main()
