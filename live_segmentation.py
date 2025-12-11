import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ==============================
#       USER SETTINGS
# ==============================

CHECKPOINT_PATH = "unet_checkpoint.pt"
NUM_CLASSES = 4   # 0=background, 1=oil, 2=soap, 3=honey
USE_GPU = True    # set False if you want CPU only
VOLUME = 36.191147369  # (Approximately) Vial Volume (cm^3)

# ==============================
#       MODEL DEFINITION
# (must match your training script)
# ==============================

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

        self.u3 = DoubleConv(256 + 512, 256)
        self.u2 = DoubleConv(128 + 256, 128)
        self.u1 = DoubleConv(64 + 128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.d1(x);    p1 = self.pool(c1)
        c2 = self.d2(p1);   p2 = self.pool(c2)
        c3 = self.d3(p2);   p3 = self.pool(c3)
        c4 = self.d4(p3)

        up3 = self.u3(torch.cat([
            F.interpolate(c4, scale_factor=2, mode="bilinear", align_corners=False),
            c3
        ], dim=1))

        up2 = self.u2(torch.cat([
            F.interpolate(up3, scale_factor=2, mode="bilinear", align_corners=False),
            c2
        ], dim=1))

        up1 = self.u1(torch.cat([
            F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False),
            c1
        ], dim=1))

        return self.final(up1)


# ==============================
#       TRANSFORMS
# (match training transform)
# ==============================

img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ==============================
#       COLOUR MAP
# ==============================

# BGR colours for OpenCV overlay
CLASS_COLOURS = {
    0: (0, 0, 0),         # background - black (invisible)
    1: (255, 0, 0),       # oil - blue
    2: (0, 255, 0),       # soap - green
    3: (0, 0, 255),       # honey - red
}


def mask_to_colour(mask):
    """
    mask: (H, W) with integer class IDs
    returns: (H, W, 3) uint8 colour image (BGR)
    """
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, bgr in CLASS_COLOURS.items():
        colour[mask == cls_id] = bgr
    return colour


# ==============================
#       MAIN LIVE LOOP
# ==============================

def main():
    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    # Load model
    model = UNet(NUM_CLASSES).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. "
                                f"Train with TASK='seg' first.")
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Open default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Keep original size for overlay
            orig_h, orig_w = frame.shape[:2]

            # Convert BGR -> RGB -> PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Transform and add batch dimension
            inp = img_transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)          # [1, C, 256, 256]
                pred = logits.argmax(1)[0]   # [256, 256]
                pred_np = pred.cpu().numpy()

            # Colour mask, resize back to original size
            colour_mask = mask_to_colour(pred_np)
            colour_mask = cv2.resize(colour_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Blend original frame and mask
            alpha = 0.4  # transparency of mask
            blended = cv2.addWeighted(frame, 1.0, colour_mask, alpha, 0)

            # Show result
            cv2.imshow("Live Segmentation (press 'q' to quit)", blended)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
