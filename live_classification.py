import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np

MODEL_PATH = "classifier_checkpoint.pt"

CLASS_NAMES = ["oil", "soap", "honey", "empty"]
CONF_THRESHOLD = 0.60

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


class ClassificationEngine:
    def __init__(self):
        model = resnet18()
        model.fc = torch.nn.Linear(512, len(CLASS_NAMES))

        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)

        model.eval()
        self.model = model

    # ------------------------------------------------------------
    #  INTERNAL inference helper
    # ------------------------------------------------------------
    def _predict_logits(self, frame):
        """Preprocess frame → logits (before softmax)."""
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)

        return logits

    # ------------------------------------------------------------
    #  MAIN classification method (top-1)
    # ------------------------------------------------------------
    def classify(self, frame):
        logits = self._predict_logits(frame)
        probs = F.softmax(logits, dim=1)

        pred_idx = probs.argmax(1).item()
        conf = probs[0, pred_idx].item()

        if conf < CONF_THRESHOLD:
            return ("none", conf)

        return (CLASS_NAMES[pred_idx], conf)

    # ------------------------------------------------------------
    #  PROBABILITY DISTRIBUTION (for UI confidence panel)
    # ------------------------------------------------------------
    def predict_proba(self, frame):
        logits = self._predict_logits(frame)
        probs = F.softmax(logits, dim=1)[0]

        return probs.cpu().numpy()
