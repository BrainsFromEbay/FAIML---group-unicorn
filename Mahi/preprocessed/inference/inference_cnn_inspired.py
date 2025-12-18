import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
from PIL import Image

# Define constants
MODEL_PATH = "Mahi/preprocessed/models/best_cnn_inspired.pth"
TEST_DIR = "custom_test"
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InspiredCNN(nn.Module):
    def __init__(self):
        super(InspiredCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

def preprocess_image(image_path):
    # Match Friend's processing logic, but using OpenCV or PIL
    # Friend used PIL:
    # image = Image.open(image_path).convert("L")
    # image = image.resize((28, 28), Image.BILINEAR)
    # image = np.array(image)
    # if image.mean() > 127: image = 255 - image
    # image = image.astype(np.float32) / 255.0
    # image = (image - 0.5) / 0.5
    
    # We use 32x32
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        image = np.array(image)
        
        # Auto-Invert
        if image.mean() > 127:
            image = 255 - image
            
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # Tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    model = InspiredCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results_path = "Mahi/preprocessed/inference/cnn_inspired_results.md"
    with open(results_path, "w") as f:
        f.write("# CNN Inspired Results\n")
        f.write("| Filename | Prediction | Confidence |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        for filename in files:
            filepath = os.path.join(TEST_DIR, filename)
            input_tensor = preprocess_image(filepath)
            
            if input_tensor is None:
                continue
                
            input_tensor = input_tensor.to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
            pred_label = predicted.item()
            conf_score = confidence.item() * 100
            
            print(f"{filename:<20} | {pred_label:<10} | {conf_score:.1f}%")
            f.write(f"| {filename} | {pred_label} | {conf_score:.1f}% |\n")
            
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
