import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms

MODEL_PATH = "Mahi/src/models/CNN_mnist.pth"
TEST_DIR = "custom_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FinalCNN(nn.Module):
    def __init__(self):
        super(FinalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image(image_path):    
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
            
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5

        
        img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
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

    model = FinalCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results_path = "Mahi/preprocessed/inference/cnn_final_results.md"
    with open(results_path, "w") as f:
        f.write("# Final CNN (MNIST) Results\n")
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
