import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_DEFAULT = "Mahi/src/models/MLP_pickle.pth"
IMG_SIZE = 32

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

def load_model(model_path=MODEL_PATH_DEFAULT):
    if not os.path.exists(model_path):
        alternative = os.path.join("Mahi", "src", "models", "MLP_pickle.pth")
        if os.path.exists(alternative):
            model_path = alternative
            
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    print(f"Loading model from {model_path}...")
    
    if 'SimpleMLP' not in sys.modules['__main__'].__dict__:
        sys.modules['__main__'].SimpleMLP = SimpleMLP

    try:
        model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        model_data = torch.load(model_path, map_location=DEVICE)
    except Exception as e:
        print(f"Error calling torch.load: {e}")
        return None

    if isinstance(model_data, nn.Module):
        model = model_data
    elif isinstance(model_data, dict):
        model = SimpleMLP().to(DEVICE)
        model.load_state_dict(model_data)
    else:
        print("Unknown model format.")
        return None
    
    model.eval()
    return model

def predict_single(model, image_path):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None, 0.0
        
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
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

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    correct_count = 0
    total_count = 0

    for filename in sorted(files):
        filepath = os.path.join(folder, filename)
        pred, conf = predict_single(model, filepath)
        
        if pred is None:
            continue
            
        print(f"{filename:<20} | {pred:<10} | {conf:.1f}%")

        try:
            base_name = filename.split('(')[0]
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
