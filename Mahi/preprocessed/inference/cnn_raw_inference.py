import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms

# Use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the exact same CNN model architecture from the training script
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

# 2. Define the exact same image transform as in the training script
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 3. Load the trained model
model_path = 'Mahi/raw_image/CNN_from_raw_image.pth'
model = SimpleCNN(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# 4. Preprocessing and Prediction Loop
folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Found {len(files)} images in '{folder}'.\n")
print(f"Using model: {model_path}")
print("-" * 40)

for f in sorted(files):
    # Get the expected digit from the filename
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    # --- Image Preprocessing ---
    # This pipeline combines manual preprocessing (inversion, thresholding)
    # with the transforms from the training script.
    img_path = os.path.join(folder, f)
    
    # Read in grayscale
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert colors (to match common digit datasets: white digit, black background)
    img_np = 255 - img_np

    # Binarize the image to remove noise
    img_np[img_np < 50] = 0
    
    # Convert to PIL Image to use torchvision transforms
    image_pil = Image.fromarray(img_np)

    # Apply the resize and ToTensor transforms
    # ToTensor converts the image to (C, H, W) and scales to [0.0, 1.0]
    img_tensor = transform(image_pil)

    # Add the batch dimension: (C, H, W) -> (N, C, H, W)
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # --- Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    print(f"File: {f}")
    print(f"Expected digit (from filename): {expected}")
    print(f"Predicted digit: {pred}")
    print(f"Match: {'Yes' if expected == pred else 'No'}")
    print("-" * 40)

print("\nInference complete.")
