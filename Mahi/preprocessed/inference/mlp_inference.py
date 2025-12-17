import torch
import numpy as np
from PIL import Image
import os
import torch.nn as nn
import cv2 # Import cv2

class SimpleMLP(nn.Module):
    def __init__(self, input_size=32*32, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleMLP()
model.load_state_dict(torch.load('Mahi/preprocessed/models/best_mlp_model.pth', map_location=torch.device('cpu')))
model.eval()

# Folder path
folder = 'custom_test'

files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Found {len(files)} images in '{folder}'.\n")

for f in sorted(files):
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    img_path = os.path.join(folder, f)
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    img_np = cv2.resize(img_np, (32, 32))

    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    img_flat = img_np.reshape(-1) / 255.0

    input_tensor = torch.tensor(img_flat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    print(f"File: {f}")
    print(f"Expected digit (from filename): {expected}")
    print(f"Predicted digit: {pred}")
    print(f"Match: {'Yes' if expected == pred else 'No'}")
    print("-" * 40)

print("\nTesting complete.")