import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

# Define constants
# Note: Training script saves to "Mahi/preprocessed/models/best_cnn_improved.pth"
MODEL_PATH = "Mahi/preprocessed/models/cnn_full_improved.pth"
TEST_DIR = "custom_test"
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Must match ImprovedDigitCNN in CNN_improved.py) ---
class ImprovedDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedDigitCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            # Global Average Pooling: 8x8 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            
            # Much smaller dense layer
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), 
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image(image_path):
    """
    Reads image, inverts (if needed), resizes, cleans, and normalizes.
    """
    # 1. Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2. Invert colors (White BG -> Black BG)
    img = 255 - img
    
    # 3. Resize to 32x32
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 4. Apply Cleaning Threshold (pixels < 50 -> 0)
    img[img < 50] = 0

    # 5. Normalize (0-1) and Shape
    img_normalized = img.astype(np.float32) / 255.0
    
    # Convert to tensor: (H, W) -> (1, H, W) -> (1, 1, H, W) for batch
    img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = ImprovedDigitCNN(num_classes=10).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        # In case we saved the full model (though script saves state_dict)
        # We need weights_only=False because the model class is defined in __main__ of training script
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    model.eval()
    return model

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found yet: {MODEL_PATH}")
        print("Please wait for training to complete.")
        return

    model = load_model()
    print(f"Model loaded on {DEVICE}")
    print("-" * 40)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 40)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    
    results_path = "Mahi/preprocessed/inference/cnn_results_table.md"
    with open(results_path, "w") as f:
        f.write("# CNN Improved Results\n")
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
