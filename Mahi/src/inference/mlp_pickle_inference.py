import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

MODEL_PATH = "Mahi/src/models/MLP_pickle.pth"
TEST_DIR = "custom_test"
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SimpleMLP(nn.Module):
    def __init__(self, input_size=32*32, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = 255 - img
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    img[img < 50] = 0
    img_flat = img.reshape(-1)
    img_normalized = img_flat.astype(np.float32) / 255.0
    
    return torch.tensor(img_normalized).unsqueeze(0) 

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleMLP().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("Trying to load full model object...")
        model = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model.eval()
    return model

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return

    model = load_model()
    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in files:
        filepath = os.path.join(TEST_DIR, filename)
        input_tensor = preprocess_image(filepath)
        
        if input_tensor is None:
            print(f"{filename:<20} | Error reading file")
            continue
            
        input_tensor = input_tensor.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        pred_label = predicted.item()
        conf_score = confidence.item() * 100
        
        print(f"{filename:<20} | {pred_label:<10} | {conf_score:.1f}%")

if __name__ == "__main__":
    main()
