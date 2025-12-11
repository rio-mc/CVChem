# ---------------------------------------------------------------------
# Segmentation Trainer
# ---------------------------------------------------------------------
# This script trains and evaluates a semantic segmentation model for
# vial contents. It delegates all training, validation, and plotting
# responsibilities to the unified trainer.py module.
#
# Scientific workflow mirrored:
#   - data preparation and partitioning
#   - model definition (encoder–decoder)
#   - centralised training loop (from trainer.py)
#   - validation pixel accuracy (computed inside trainer.py)
#   - qualitative inspection of segmentation output
# ---------------------------------------------------------------------

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Unified trainer module
from trainer import train_loop, plot_training_curves


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEGROOT = "all_data_seg/"
SEG_FILE = "SEG_LABELS.txt"

SEG_NUM_CLASSES = 5  # background + 4 semantic classes

BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15


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
# Utility Functions
# ---------------------------------------------------------------------

def inspect_all_masks(root):
    """
    Report unique label values appearing in each mask image.
    Used to verify consistency of annotation practices.
    """
    print("\n=== MASK VALUE SUMMARY ===")
    for f in sorted(os.listdir(root)):
        if f.endswith("_mask.png"):
            arr = np.array(Image.open(os.path.join(root, f)))
            unique = np.unique(arr)
            print(f"{f}: {unique}")
    print("=== END SUMMARY ===\n")


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def load_label_file(path):
    """
    SEG_FILE lists only the input image names.
    Mask images contain the true semantic labels.
    """
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            name, _ = line.strip().split(",")
            mapping[name] = 0
    return mapping


def autosplit(root, labels_dict):
    """
    Partition the dataset into train/validation/test splits.
    Ensures that no image appears in multiple sets.
    """
    files = sorted([
        f for f in os.listdir(root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
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


# ---------------------------------------------------------------------
# Image / Mask Transforms
# ---------------------------------------------------------------------

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

mask_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)
])

# Colour-coded interpretation of semantic labels
CLASS_COLOURS = {
    0: (0,   0,   0),
    1: (180, 180, 180),
    2: (0,   0,   255),
    3: (255, 0,   0),
    4: (0, 255,   0),
}

CLASS_NAMES = {
    0: "background",
    1: "empty",
    2: "honey",
    3: "oil",
    4: "soap",
}

def mask_to_colour(mask):
    """
    Convert integer-valued mask into an RGB image for visualisation.
    """
    mask = np.asarray(mask)
    h, w = mask.shape
    col = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, bgr in CLASS_COLOURS.items():
        col[mask == cid] = bgr
    return col


def undo_normalisation(img_t):
    """
    Undo ImageNet normalisation.
    """
    img = img_t.clone()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def draw_title(ax, text):
    """
    Draw text with stroke outline for robust legibility.
    """
    ax.set_title(
        text,
        fontsize=14,
        color="white",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )


# ---------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path  = os.path.join(SEGROOT, name)
        mask_path = os.path.splitext(img_path)[0] + "_mask.png"

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img  = img_transform(img)
        mask = torch.from_numpy(np.array(mask_transform(mask))).long()

        return img, mask


# ---------------------------------------------------------------------
# Model Definition: ResNet18 Encoder + UNet Decoder
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)


class UNetResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Encoder
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4

        # Decoder
        self.up4 = DoubleConv(512 + 256, 256)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64,  64)
        self.up1 = DoubleConv(64  + 64,  64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)

        u4 = self.up4(torch.cat([
            F.interpolate(c5, scale_factor=2, mode="bilinear", align_corners=False),
            c4
        ], dim=1))
        u3 = self.up3(torch.cat([
            F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False),
            c3
        ], dim=1))
        u2 = self.up2(torch.cat([
            F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False),
            c2
        ], dim=1))
        u1 = self.up1(torch.cat([
            F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False),
            c1
        ], dim=1))

        return self.final(u1)


# ---------------------------------------------------------------------
# Visualisation of Predictions
# ---------------------------------------------------------------------

def show_segmentation_examples(model, loader, device, max_images=6):
    """
    Shows: raw image, ground truth mask, predicted mask, and overlay.
    """
    model.eval()

    def classes_in_mask(mask_np):
        unique = np.unique(mask_np)
        return [CLASS_NAMES[int(k)] for k in unique]

    total_samples = len(loader.dataset)
    rows = min(max_images, total_samples)

    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    idx_row = 0

    with torch.no_grad():
        for imgs, masks in loader:
            if idx_row >= rows:
                break

            imgs = imgs.to(device)
            logits = model(imgs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            preds = logits.argmax(1).cpu().numpy()
            imgs_cpu  = imgs.cpu()
            masks_cpu = masks.numpy()

            bs = imgs_cpu.shape[0]

            for i in range(bs):
                if idx_row >= rows:
                    break

                img_disp  = undo_normalisation(imgs_cpu[i])
                gt_mask   = masks_cpu[i]
                pred_mask = preds[i]

                gt_classes   = classes_in_mask(gt_mask)
                pred_classes = classes_in_mask(pred_mask)
                per_pixel_acc = (pred_mask == gt_mask).mean()

                # -------------------------------------------------
                # Column 1: Original
                # -------------------------------------------------
                ax1 = axes[idx_row, 0]
                ax1.imshow(img_disp)
                ax1.axis("off")
                ax1.text(
                    0.02, 0.12,
                    f"GT: {gt_classes}\nPred: {pred_classes}",
                    color="white", fontsize=10, transform=ax1.transAxes,
                    ha="left", va="top",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")]
                )

                # -------------------------------------------------
                # Column 2: Ground Truth
                # -------------------------------------------------
                ax2 = axes[idx_row, 1]
                ax2.imshow(mask_to_colour(gt_mask))
                draw_title(ax2, "Ground Truth")
                ax2.axis("off")

                # -------------------------------------------------
                # Column 3: Prediction
                # -------------------------------------------------
                ax3 = axes[idx_row, 2]
                ax3.imshow(mask_to_colour(pred_mask))
                draw_title(ax3, "Prediction")
                ax3.axis("off")

                ax3.text(
                    0.02, 0.05,
                    f"Acc: {per_pixel_acc:.3f}",
                    color="white", fontsize=10,
                    transform=ax3.transAxes,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")]
                )

                # -------------------------------------------------------------------------
                # Column 4: Overlay
                #     Reveals boundary errors, leakage into background, and class confusion
                #     more clearly than isolated masks.
                # -------------------------------------------------------------------------

                ax4 = axes[idx_row, 3]

                coloured_pred = mask_to_colour(pred_mask) / 255.0
                overlay = (0.85 * img_disp) + (0.15 * coloured_pred)

                ax4.imshow(overlay)
                draw_title(ax4, "Overlay")
                ax4.axis("off")

                idx_row += 1

    plt.show()


# ---------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------

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

    model = UNetResNet18(SEG_NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------------
    # Centralised training loop (from trainer.py)
    # -----------------------------------------------------------------
    log = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimiser=optimiser,
        epochs=EPOCHS,
        task_type="segmentation"
    )

    plot_training_curves(log, metric_name="Validation Pixel Accuracy")

    # -----------------------------------------------------------------
    # Qualitative inspection on test data
    # -----------------------------------------------------------------
    show_segmentation_examples(model, test_loader, device)

    save_checkpoint(model, "segmentation_checkpoint.pt")


if __name__ == "__main__":
    inspect_all_masks(SEGROOT)
    main()
