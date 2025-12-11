import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


CHECKPOINT_PATH = "unet_checkpoint.pt"

# Classes:
# 0 = background
# 1 = vial interior
# 2 = oil
# 3 = soap
# 4 = honey
NUM_CLASSES = 5

USE_GPU = True
VIAL_VOLUME_ML = 36.191147369


# ------------------------------
# U-Net model
# ------------------------------

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

        self.pool = nn.MaxPool2d(2)

        self.u3 = DoubleConv(256 + 512, 256)
        self.u2 = DoubleConv(128 + 256, 128)
        self.u1 = DoubleConv(64 + 128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x); p1 = self.pool(c1)
        c2 = self.d2(p1); p2 = self.pool(c2)
        c3 = self.d3(p2); p3 = self.pool(c3)
        c4 = self.d4(p3)

        u3 = self.u3(torch.cat([F.interpolate(c4, scale_factor=2), c3], dim=1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2), c2], dim=1))
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2), c1], dim=1))

        return self.final(u1)


# ------------------------------
# Transforms
# ------------------------------

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


CLASS_COLOURS = {
    0: (0, 0, 0),
    1: (180, 180, 180),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (0, 0, 255),
}


def mask_to_colour(mask):
    h, w = mask.shape
    col = np.zeros((h, w, 3), np.uint8)
    for cls, bgr in CLASS_COLOURS.items():
        col[mask == cls] = bgr
    return col


# ------------------------------
# Volume calculation
# ------------------------------

def compute_volume(pred_mask, vial_volume_ml):
    vial_pix = np.where(pred_mask == 1)
    if vial_pix[0].size == 0:
        return 0.0, 0.0

    vial_top = vial_pix[0].min()
    vial_bottom = vial_pix[0].max()
    vial_h = vial_bottom - vial_top

    liquid_pix = np.where(pred_mask >= 2)
    if liquid_pix[0].size == 0:
        return 0.0, 0.0

    liquid_bottom = liquid_pix[0].max()
    liquid_h = max(0, liquid_bottom - vial_top)

    frac = np.clip(liquid_h / vial_h, 0, 1)
    return frac, frac * vial_volume_ml


# ------------------------------
# Live loop
# ------------------------------

def main():

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print("Using:", device)

    model = UNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Segmentation model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = img_transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(1)[0].cpu().numpy()

        mask_col = mask_to_colour(pred)
        mask_col = cv2.resize(mask_col, (w, h), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(frame, 1.0, mask_col, 0.4, 0)

        frac, vol = compute_volume(pred, VIAL_VOLUME_ML)

        cv2.putText(blended, f"{frac*100:.1f}%  {vol:.2f} mL",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        cv2.imshow("Live Segmentation + Volume", blended)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
