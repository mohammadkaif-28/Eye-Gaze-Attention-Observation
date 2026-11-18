"""Small single-frame baseline; not the original temporal Gaze360 architecture."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 128
TRANSFORM = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])


class GazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (24, 48, 96, 128):
            layers += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                       nn.GroupNorm(8, width), nn.ReLU()]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(128, 3))

    def forward(self, x):
        return nn.functional.normalize(self.head(self.features(x)), dim=-1, eps=1e-6)


def angular_error(prediction, target):
    dot = (prediction * target).sum(-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(dot))


def vector_direction(vector, threshold=15):
    x, y, z = map(float, vector)
    norm = math.sqrt(x*x+y*y+z*z)
    if not math.isfinite(norm) or norm < 1e-6:
        return "indeterminate"
    # Gaze360: x left, y up, z away, all in the camera/viewer's frame.
    yaw = math.degrees(math.atan2(-x, -z))
    pitch = math.degrees(math.atan2(-y, math.hypot(x,z)))
    if z >= 0 or abs(yaw) > 60 or abs(pitch) > 50:
        return "indeterminate"
    if max(abs(yaw),abs(pitch)) < threshold:
        return "towards camera"
    if abs(yaw) >= abs(pitch):
        return "right" if yaw > 0 else "left"
    return "down" if pitch > 0 else "up"


def load_checkpoint(path):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if data.get("architecture") != "gazenet-v1" or data.get("coordinates") != "gaze360":
        raise ValueError("Unsupported checkpoint. Use a best.pt produced by this project's training script.")
    model = GazeNet()
    model.load_state_dict(data["model"])
    return model, data


class Predictor:
    def __init__(self, checkpoint):
        self.model, _ = load_checkpoint(checkpoint)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, bgr):
        image = Image.fromarray(np.ascontiguousarray(bgr[:,:,::-1]))
        prediction = self.model(TRANSFORM(image).unsqueeze(0))[0].numpy()
        return vector_direction(prediction)
