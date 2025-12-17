import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
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
    image = Image.open(image_path).convert("L")

    image = image.resize((28, 28), Image.BILINEAR)

    image = np.array(image)

    if image.mean() > 127:
        image = 255 - image

    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)
    
    return image

def predict_image(image_path, model, device, show_image=True):
    model.eval()

    image = preprocess_image(image_path)

    image = image.to(device)

    if show_image:
        plt.imshow(image.cpu().squeeze(), cmap="gray")
        plt.title(f"Input: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1)[0]
        confidence = probabilities[prediction].item() * 100

    return prediction, confidence

def predict_all_digits(folder_path, model, device, show_images=False):
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))

    results = []

    if not image_paths:
        print("No images found in the specified folder.")
        return results
    
    print (f"Found {len(image_paths)} images.")

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        prediction, confidence = predict_image(image_path, model, device, show_image=show_images)

        results.append({
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence
        })

        print(f"{filename:30s} => Predicted: {prediction}, Confidence: {confidence:.2f}%")

    return results

def load_model(model_path="cnn_mnist.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded saved model")

    return model

if __name__ == "__main__":
    model = load_model("cnn_mnist.pth")

    folder_path = r"letterRecognition/digits"
    results = predict_all_digits(folder_path, model, device, show_images=False)