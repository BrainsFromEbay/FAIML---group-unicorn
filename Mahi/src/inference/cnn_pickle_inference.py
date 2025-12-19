import torch
import torch.nn as nn
import numpy as np
import cv2
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model_path = 'Mahi/src/models/CNN_pickle.pth'
model = torch.load(model_path, map_location=DEVICE, weights_only=False)
model.eval()
folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Model loaded on {DEVICE}")
print("-" * 55)
print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
print("-" * 55)

correct_count = 0
total_count = 0

for f in sorted(files):
    name = f.split('.')[0].strip()
    
    img_path = os.path.join(folder, f)
    
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        continue

    img_np = cv2.resize(img_np, (32, 32))

    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32)

    input_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred = predicted.item()
    conf_score = confidence.item() * 100

    print(f"{f:<20} | {pred:<10} | {conf_score:.1f}%")

    # Calculate accuracy
    try:
        base_name = f.split('(')[0]
        expected = int(base_name.split('.')[0])
        
        if expected == pred:
            correct_count += 1
        total_count += 1
    except ValueError:
        pass

if total_count > 0:
    accuracy = (correct_count / total_count) * 100
    print("-" * 55)
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
    print("-" * 55)

