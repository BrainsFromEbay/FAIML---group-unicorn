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

print(f"Found {len(files)} images in '{folder}'.\n")
print(f"Using model: {model_path}")
print("-" * 40)

for f in sorted(files):
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    img_path = os.path.join(folder, f)
    
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    img_np = 255 - img_np
    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    image_pil = Image.fromarray(img_np)

    img_tensor = transform(image_pil)

    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    print(f"File: {f}")
    print(f"Expected digit (from filename): {expected}")
    print(f"Predicted digit: {pred}")
    print(f"Match: {'Yes' if expected == pred else 'No'}")
    print("-" * 40)

print("\nInference complete.")
