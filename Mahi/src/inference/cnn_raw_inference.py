import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_DEFAULT = 'Mahi/src/models/CNN_raw.pth'

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

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def load_model(model_path=MODEL_PATH_DEFAULT):
    if not os.path.exists(model_path):
        alternative = os.path.join("Mahi", "src", "models", "CNN_raw.pth")
        if os.path.exists(alternative):
            model_path = alternative
            
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    model = SimpleCNN(num_classes=10).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None
        
    model.eval()
    return model

def preprocess_image(img_path):
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        return None
    
    img_np = 255 - img_np
    img_np[img_np < 50] = 0
    
    image_pil = Image.fromarray(img_np)
    img_tensor = transform(image_pil)
    input_tensor = img_tensor.unsqueeze(0)
    return input_tensor

def predict_single(model, image_path):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None, 0.0
        
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item() * 100

def main():
    folder = 'custom_test'
    if not os.path.exists(folder):
        print(f"Folder {folder} not found")
        return

    model = load_model()
    if model is None:
        return

    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    correct_count = 0
    total_count = 0

    for f in sorted(files):
        img_path = os.path.join(folder, f)
        pred, conf = predict_single(model, img_path)
        
        if pred is None:
            continue

        print(f"{f:<20} | {pred:<10} | {conf:.1f}%")

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

if __name__ == "__main__":
    main()
