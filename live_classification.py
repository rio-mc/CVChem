import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np

MODEL_PATH = "classifier_checkpoint.pt"

labels = ["oil", "soap", "honey", "empty"]
CONF_THRESHOLD = 0.60

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ClassificationEngine:
    def __init__(self):
        model = resnet18()
        model.fc = torch.nn.Linear(512, 4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        self.model = model

    def classify(self, frame):
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        pred = probs.argmax(1).item()
        conf = probs[0, pred].item()

        if conf < CONF_THRESHOLD:
            return ("none", conf)
        return (labels[pred], conf)
