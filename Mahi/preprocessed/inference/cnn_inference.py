import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# Use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the exact same CNN model architecture from the training script
class SimpleDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Load the trained model
# The training script saves the entire model using pickle, so we must set weights_only=False
# as per the new PyTorch 2.6+ security default. This is safe since the model was trained
# within this project.
model_path = 'Mahi/preprocessed/models/CNN_digit_full.pth'
model = torch.load(model_path, map_location=DEVICE, weights_only=False)
model.eval()


# 3. Preprocessing and Prediction Loop
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
    # This pipeline is based on the one that worked for the MLP model
    # trained on the same pickled dataset.
    img_path = os.path.join(folder, f)
    
    # Read in grayscale
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 32x32, which is the input size for the pickled data
    img_np = cv2.resize(img_np, (32, 32))

    # Invert colors (black background, white digit)
    img_np = 255 - img_np

    # Binarize the image to remove noise
    img_np[img_np < 50] = 0
    
    # Scale pixel values to [0.0, 1.0]
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32)

    # Add batch and channel dimensions: (H, W) -> (N, C, H, W)
    # The CNN expects a 4D tensor: (1, 1, 32, 32)
    input_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

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
