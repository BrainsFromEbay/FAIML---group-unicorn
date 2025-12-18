import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

# Define constants
MODEL_PATH = "Mahi/preprocessed/models/best_mlp_improved.pth"
TEST_DIR = "custom_test"
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Must match the trained model in MLP_improved.py) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=32*32, num_classes=10):
        super(SimpleMLP, self).__init__()
        # Reduced architecture complexity as requested
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
    """
    Reads image, inverts (if needed), resizes, cleans, and normalizes
    to match training data (black background, white digits).
    """
    # 1. Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2. Invert colors (User said inputs have White BG, training had Black BG)
    # White (255) -> Black (0)
    img = 255 - img
    
    # 3. Resize to 32x32
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 4. Apply Cleaning Threshold (from pickle_clean.py)
    # pickle_clean.py sets pixels < 50 to 0. 
    # Since we inverted, background noise (originally near 255, now near 0) will be cleaned.
    img[img < 50] = 0

    # 5. Flatten and Normalize
    # SimpleMLP expects flat input. 
    img_flat = img.reshape(-1)
    img_normalized = img_flat.astype(np.float32) / 255.0
    
    return torch.tensor(img_normalized).unsqueeze(0) # Add batch dim -> (1, 1024)

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleMLP().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        # Fallback if full model was saved instead of state_dict or path mismatch
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
    print("-" * 40)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 40)

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
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        pred_label = predicted.item()
        conf_score = confidence.item() * 100
        
        print(f"{filename:<20} | {pred_label:<10} | {conf_score:.1f}%")

if __name__ == "__main__":
    main()
