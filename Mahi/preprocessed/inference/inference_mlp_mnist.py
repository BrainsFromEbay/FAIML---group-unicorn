import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# Define constants
MODEL_PATH = "Mahi/preprocessed/models/mlp_mnist_best.pth"
TEST_DIR = "custom_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

def preprocess_image(image_path):
    # Same logic as CNN_Final_MNIST (Friend's Logic)
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        
        # Auto-invert check
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
            
        # Normalization [-1, 1]
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        
        # To Tensor (1, 1, 28, 28)
        img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0) 
        return img_tensor
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found (yet): {MODEL_PATH}")
        return

    model = MNIST_MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"MLP Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results_path = "Mahi/preprocessed/inference/mlp_mnist_results.md"
    with open(results_path, "w") as f:
        f.write("# MLP (MNIST) Results\n")
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
