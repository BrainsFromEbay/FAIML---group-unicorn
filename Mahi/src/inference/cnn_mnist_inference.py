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
        

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        # Catch standard relative path issue
        if os.path.exists(os.path.join("Mahi", "src", "models", "CNN_mnist.pth")):
             model_path = os.path.join("Mahi", "src", "models", "CNN_mnist.pth")
        
    if not os.path.exists(model_path):
         print(f"Model file not found: {model_path}")
         return None

    model = FinalCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict_single(model, image_path):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None, None
        
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    return predicted.item(), confidence.item() * 100

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    model = load_model()
    if model is None:
        return

    print(f"Model loaded on {DEVICE}")
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)


    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    correct_count = 0
    total_count = 0

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
        # Calculate accuracy
        try:
            # Handle standard "0.png" and noisy "0(1).png"
            # Split by '(' first to ignore noise suffix, then split by '.'
            base_name = filename.split('(')[0]
            expected = int(base_name.split('.')[0])
            
            if expected == pred_label:
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
