import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
model_path = 'Mahi/src/models/CNN_raw.pth'
model = SimpleCNN(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Model loaded on {DEVICE}")
print("-" * 55)
print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
print("-" * 55)

for f in sorted(files):
    name = f.split('.')[0].strip()

    img_path = os.path.join(folder, f)
    
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        continue
    
    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    image_pil = Image.fromarray(img_np)

    img_tensor = transform(image_pil)

    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred = predicted.item()
    conf_score = confidence.item() * 100

    print(f"{f:<20} | {pred:<10} | {conf_score:.1f}%")

