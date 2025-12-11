import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import pyrealsense2 as rs
from scipy.ndimage import label as cc_label


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "unet_checkpoint.pt"

# 0 background, 1 vial interior (glass/empty), 2 oil, 3 soap, 4 honey
NUM_CLASSES = 5
USE_GPU = True
VIAL_VOLUME_ML = 36.191147369   # known physical vial volume

CLASS_NAMES = {
    0: "background",
    1: "vial interior",
    2: "honey",
    3: "oil",
    4: "soap"
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

        u4 = self.up4(torch.cat(
            [F.interpolate(c5, scale_factor=2, mode="bilinear", align_corners=False), c4],
            dim=1
        ))
        u3 = self.up3(torch.cat(
            [F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False), c3],
            dim=1
        ))
        u2 = self.up2(torch.cat(
            [F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), c2],
            dim=1
        ))
        u1 = self.up1(torch.cat(
            [F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), c1],
            dim=1
        ))

        return self.final(u1)


# ============================================================
# TRANSFORMS
# ============================================================

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def mask_to_colour(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), np.uint8)
    for cid, col in CLASS_COLOURS.items():
        img[mask == cid] = col
    return img


# ============================================================
# PER-VIAL ANALYSIS (Connected Components)
# ============================================================
""" Dominant liquid visualisation
def analyse_vials(pred_mask, vial_volume_ml):
    """
    #Split each vial into separate connected components in the
    #segmentation space (256x256), compute the dominant liquid
    #class, fill %, and volume for each.

    #We treat any non-background pixel (class != 0) as part of
    #a vial; liquid classes (>= 2) are searched within each
    #connected component.
"""
    # Everything except background is considered part of a vial object
    vial_mask = (pred_mask != 0).astype(np.uint8)

    labeled, n = cc_label(vial_mask)

    results = []

    for vial_id in range(1, n + 1):
        coords = np.column_stack(np.where(labeled == vial_id))
        if coords.size == 0:
            continue

        ys, xs = coords[:, 0], coords[:, 1]
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        vial_h = ymax - ymin + 1
        if vial_h <= 1:
            continue

        # Identify liquid pixels inside this vial (classes 2, 3, 4)
        inside_vial = (labeled == vial_id)
        liquid_mask = (pred_mask >= 2) & inside_vial
        liquid_pixels = pred_mask[liquid_mask]

        if liquid_pixels.size == 0:
            liquid_class = None
            percentage = 0.0
            volume_ml = 0.0
        else:
            # Dominant liquid class in this vial
            classes, counts = np.unique(liquid_pixels, return_counts=True)
            liquid_class = int(classes[np.argmax(counts)])

            liquid_coords = np.column_stack(np.where(liquid_mask))
            liquid_ys = liquid_coords[:, 0]

            # Vial bottom and liquid top in image coordinates
            vial_bottom = ymax
            liquid_top = liquid_ys.min()

            filled_h = vial_bottom - liquid_top + 1
            # Clamp to [0, vial_h] in case of odd predictions
            filled_h = max(0, min(filled_h, vial_h))

            percentage = float(filled_h) / float(vial_h)
            percentage = float(np.clip(percentage, 0.0, 1.0))
            volume_ml = percentage * vial_volume_ml

        results.append({
            # BBox in segmentation (256x256) coordinates
            "bbox": (xmin, ymin, xmax, ymax),
            "liquid_class": liquid_class,
            "percentage": percentage,
            "volume_ml": volume_ml
        })

    return results
"""
def analyse_vials(pred_mask, vial_volume_ml):
    """
    Per-vial, multi-liquid analysis.

    For each connected component of non-background pixels (i.e. a vial),
    we compute:
        - overall vial bounding box
        - per-liquid class vertical extent (height within vial)
        - per-liquid percentage of vial height
        - per-liquid volume (mL)

    Returns
    -------
    results : list of dict
        Each element:
        {
            "bbox": (xmin, ymin, xmax, ymax),  # in 256×256 mask coords
            "layers": [
                {
                    "class_id": int,
                    "percentage": float,   # 0–1
                    "volume_ml": float,
                    "top": int,            # y in mask coords
                    "bottom": int          # y in mask coords
                },
                ...
            ]
        }
        Layers are sorted from bottom to top in image coordinates.
    """

    # Everything except background is part of some vial object
    vial_mask = (pred_mask != 0).astype(np.uint8)

    labeled, n = cc_label(vial_mask)

    results = []

    for vial_id in range(1, n + 1):
        coords = np.column_stack(np.where(labeled == vial_id))
        if coords.size == 0:
            continue

        ys, xs = coords[:, 0], coords[:, 1]
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        vial_h = ymax - ymin + 1
        if vial_h <= 1:
            continue

        inside_vial = (labeled == vial_id)

        layers = []

        # Process each liquid class independently (2, 3, 4)
        for class_id in [2, 3, 4]:
            liquid_mask = (pred_mask == class_id) & inside_vial
            liquid_coords = np.column_stack(np.where(liquid_mask))

            if liquid_coords.size == 0:
                continue

            liquid_ys = liquid_coords[:, 0]

            # Vertical extent of this liquid class within the vial
            top = liquid_ys.min()
            bottom = liquid_ys.max()

            height = bottom - top + 1
            # Clamp height to [0, vial_h]
            height = max(0, min(height, vial_h))

            percentage = float(height) / float(vial_h)
            percentage = float(np.clip(percentage, 0.0, 1.0))
            volume_ml = percentage * vial_volume_ml

            layers.append({
                "class_id": class_id,
                "percentage": percentage,
                "volume_ml": volume_ml,
                "top": int(top),
                "bottom": int(bottom)
            })

        # If no liquid classes detected, treat vial as empty
        # (you may want to drop empty vials instead)
        layers.sort(key=lambda d: d["top"], reverse=True)  # bottom first (largest y)

        results.append({
            "bbox": (xmin, ymin, xmax, ymax),
            "layers": layers
        })

    return results

# ============================================================
# LIVE REALSENSE LOOP
# ============================================================

def main():

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print("Using:", device)

    # Load segmentation model
    model = UNetResNet18(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    # -----------------------------------
    # RealSense initialisation
    # -----------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    print("RealSense stream started.")

    try:
        while True:

            frames = pipeline.wait_for_frames()
            colour_frame = frames.get_color_frame()
            if not colour_frame:
                continue

            frame = np.asanyarray(colour_frame.get_data())
            H, W = frame.shape[:2]

            # -------------------------------
            # Prepare input for model
            # -------------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inp = img_transform(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)

                # Guarantee 256×256 output
                logits = F.interpolate(
                    logits, size=(256, 256),
                    mode="bilinear", align_corners=False
                )

                pred_mask = logits.argmax(1)[0].cpu().numpy()

            # -------------------------------
            # Visual mask overlay
            # -------------------------------
            col = mask_to_colour(pred_mask)
            col = cv2.resize(col, (W, H), cv2.INTER_NEAREST)
            blended = cv2.addWeighted(frame, 1.0, col, 0.4, 0)

            # --------------------------------------------------
            # Per-vial multilayer volume analysis
            # --------------------------------------------------
            vials = analyse_vials(pred_mask, VIAL_VOLUME_ML)

            mask_h, mask_w = pred_mask.shape
            scale_x = W / float(mask_w)
            scale_y = H / float(mask_h)

            for vial in vials:

                xmin_s, ymin_s, xmax_s, ymax_s = vial["bbox"]

                # Reproject bbox into camera coordinates
                xmin = int(xmin_s * scale_x)
                xmax = int((xmax_s + 1) * scale_x)
                ymin = int(ymin_s * scale_y)
                ymax = int((ymax_s + 1) * scale_y)

                # Clamp to valid image region
                xmin = max(0, min(xmin, W - 1))
                xmax = max(0, min(xmax, W - 1))
                ymin = max(0, min(ymin, H - 1))
                ymax = max(0, min(ymax, H - 1))

                cv2.rectangle(blended, (xmin, ymin), (xmax, ymax),
                              (255, 255, 0), 2)

                layers = vial["layers"]

                # -------------------------------
                # Handle empty vial
                # -------------------------------
                if not layers:
                    text_y = max(0, ymin - 10)
                    label_text = "empty  0.0%  0.00 mL"
                    cv2.putText(blended, label_text, (xmin, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                    continue

                # -------------------------------
                # Summary line above vial
                # -------------------------------
                total_percentage = sum(layer["percentage"] for layer in layers)
                total_volume = sum(layer["volume_ml"] for layer in layers)

                summary = f"mixture  {total_percentage*100:.1f}%  {total_volume:.2f} mL"

                text_y = max(0, ymin - 12)
                cv2.putText(blended, summary, (xmin, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

                # -------------------------------
                # Per-layer information
                # -------------------------------
                line_x = xmax + 5
                line_y = ymin + 15
                line_step = 18

                for layer in layers:
                    cid = layer["class_id"]
                    pct = layer["percentage"] * 100.0
                    vol = layer["volume_ml"]

                    name = CLASS_NAMES.get(cid, f"class {cid}")
                    line = f"{name}: {pct:.1f}%  {vol:.2f} mL"

                    cv2.putText(blended, line, (line_x, line_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
                    line_y += line_step

            # -------------------------------
            # Display
            # -------------------------------
            cv2.imshow("RealSense Segmentation + Per-Vial Volume", blended)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
