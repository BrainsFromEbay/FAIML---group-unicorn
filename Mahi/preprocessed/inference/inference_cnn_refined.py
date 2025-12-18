import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

# Define constants
MODEL_PATH = "Mahi/preprocessed/models/best_cnn_refined.pth"
TEST_DIR = "custom_test"
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLEAN_THRESHOLD = 50 # Reverting to 50 to see if noise was the issue with 20

class RefinedDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RefinedDigitCNN, self).__init__()
        
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
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Invert (White BG -> Black BG)
    img = 255 - img
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Cleaning with RELAXED threshold
    # This is the key change for inference
    img[img < CLEAN_THRESHOLD] = 0

    # Normalize (0-1)
    img_normalized = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = RefinedDigitCNN(num_classes=10).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        # In case we eventually save full model
        try:
             model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except:
             # Fallback if file doesn't exist yet (training ongoing)
             return None
    
    model.eval()
    return model

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    model = load_model()
    if model is None:
        print("Model file not ready yet.")
        return

    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results_path = "Mahi/preprocessed/inference/cnn_refined_results.md"
    with open(results_path, "w") as f:
        f.write("# CNN Refined Results\n")
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
