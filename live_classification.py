import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18
import numpy as np
import pyrealsense2 as rs

# ---------------------------
# Load model
# ---------------------------
model = resnet18()
model.fc = torch.nn.Linear(512, 4)
model.load_state_dict(torch.load("classifier_checkpoint.pt", map_location="cpu"))
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

labels = ["oil", "soap", "honey", "empty"]

CONF_THRESHOLD = 0.60

# ---------------------------
# RealSense setup
# ---------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# ---------------------------
# Live loop
# ---------------------------
try:
    while True:
        # Wait for a coherent frame set
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Convert to PIL image for torchvision
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(1).item()
            conf = probs[0, pred].item()

        if conf < CONF_THRESHOLD:
            text = f"No vial detected ({conf:.2f})"
            colour = (0, 0, 255)  # red
        else:
            text = f"{labels[pred]} ({conf:.2f})"
            colour = (0, 255, 0)  # green

        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2
        )

        cv2.imshow("RealSense Classification", frame)
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
