import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from scipy.ndimage import label as cc_label

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "segmentation_checkpoint.pt"
NUM_CLASSES = 5
USE_GPU = True
VIAL_VOLUME_ML = 36.191147369  # known physical vial volume

CLASS_NAMES = {
    0: "background",
    1: "vial interior",
    2: "honey",
    3: "oil",
    4: "soap",
}

CLASS_COLOURS = {
    0: (0, 0, 0),
    1: (180, 180, 180),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (0, 0, 255),
}

# ============================================================
# MODEL ARCHITECTURE — ResNet18 U-Net
# ============================================================

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


class UNetResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # If your torchvision is older and does not support weights=,
        # change the next line to: backbone = models.resnet18(pretrained=True)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4

        self.up4 = DoubleConv(512 + 256, 256)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 64, 64)

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

# ============================================================
# TRANSFORMS & VIS
# ============================================================

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def mask_to_colour(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), np.uint8)
    for cid, col in CLASS_COLOURS.items():
        img[mask == cid] = col
    return img

# ============================================================
# PER-VIAL MULTI-LAYER ANALYSIS
# ============================================================

def analyse_vials(pred_mask, vial_volume_ml):
    """
    Per-vial, multi-liquid analysis.
    Returns list of dicts:
        {
            "bbox": (xmin, ymin, xmax, ymax),
            "layers": [
                {
                    "class_id": int,
                    "percentage": float,  # 0–1 fraction of vial height
                    "volume_ml": float,
                    "top": int,
                    "bottom": int
                },
                ...
            ]
        }
    """
    vial_mask = (pred_mask != 0).astype(np.uint8)
    labelled, n = cc_label(vial_mask)
    results = []

    for vial_id in range(1, n + 1):
        coords = np.column_stack(np.where(labelled == vial_id))
        if coords.size == 0:
            continue

        ys, xs = coords[:, 0], coords[:, 1]
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        vial_h = ymax - ymin + 1

        if vial_h <= 1:
            continue

        inside_vial = (labelled == vial_id)
        layers = []

        for class_id in [2, 3, 4]:
            liquid_mask = (pred_mask == class_id) & inside_vial
            liquid_coords = np.column_stack(np.where(liquid_mask))

            if liquid_coords.size == 0:
                continue

            liquid_ys = liquid_coords[:, 0]
            top = liquid_ys.min()
            bottom = liquid_ys.max()
            height = max(0, min(bottom - top + 1, vial_h))

            percentage = float(height) / float(vial_h)
            percentage = float(np.clip(percentage, 0.0, 1.0))
            volume_ml = percentage * vial_volume_ml

            layers.append({
                "class_id": class_id,
                "percentage": percentage,
                "volume_ml": volume_ml,
                "top": int(top),
                "bottom": int(bottom),
            })

        layers.sort(key=lambda d: d["top"], reverse=True)

        results.append({
            "bbox": (xmin, ymin, xmax, ymax),
            "layers": layers
        })

    return results

# ============================================================
# SEGMENTATION ENGINE
# ============================================================

class SegmentationEngine:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
        print(f"[SegmentationEngine] Using device: {self.device}")

        self.model = UNetResNet18(NUM_CLASSES).to(self.device)

        # Load checkpoint on same device
        print(f"[SegmentationEngine] Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        # If the checkpoint was saved with DataParallel, you might need to strip 'module.' prefixes.
        self.model.load_state_dict(state)
        self.model.eval()
        print("[SegmentationEngine] Model loaded and ready")

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Takes a BGR frame (as from OpenCV), returns a 2D mask (256 x 256) of class indices.
        """
        # BGR → RGB → PIL
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        inp = img_transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(inp)
            logits = F.interpolate(
                logits,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

        pred_mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return pred_mask
