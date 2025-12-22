import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Device configuration (I really hope you have a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN architecture. Matches the training one.
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
    
# Image preprocessing
# Preprocess input image to match MNIST format:
# - Grayscale
# - Resize to 28x28
# - Normalized 
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

# Predict digit from image
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

def predict_custom_test(model, device):
    folder_path = "../custom_test"
    if not os.path.exists(folder_path):
        if os.path.exists("custom_test"):
             folder_path = "custom_test"
        elif os.path.exists("../custom_test"):
             folder_path = "../custom_test"
        else:
             print("Could not find custom_test directory!")
             return

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    true_labels = []
    pred_labels = []
    results = []

    print("\n## Prediction Matrix (Full Test Set)\n")
    print("| Image | Expected | Predicted | Confidence | Result |")
    print("| :---: | :------: | :-------: | :--------: | :----: |")

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        try:
            label = int(filename[0])
        except:
            label = -1

        prediction, confidence = predict_image(image_path, model, device, show_image=False)
        
        true_labels.append(label)
        pred_labels.append(prediction)
        
        result_icon = "✅" if label == prediction else "❌"
        print(f"| {filename} | {label} | {prediction} | {confidence:.1f}% | {result_icon} |")
        
        results.append((filename, label, prediction, confidence, result_icon))

    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n## Accuracy Scoreboard\n")
    print(f"**Accuracy: {accuracy*100:.1f}%** ({sum([1 for t,p in zip(true_labels, pred_labels) if t==p])}/{len(true_labels)})")
    
    if os.path.basename(os.getcwd()) == "Jere":
        results_dir = "results"
    else:
        results_dir = "Jere/results"
        
    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(10)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    return results, accuracy

def load_model(model_path="cnn_mnist.pth", base_path="Jere"):
    if not os.path.exists(model_path):
         alt_path = os.path.join(base_path, "cnn_mnist.pth")
         if os.path.exists(alt_path):
             model_path = alt_path
         else:
             if os.path.exists("cnn_mnist.pth"):
                 model_path = "cnn_mnist.pth"
             else:
                pass
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded saved model from {model_path}")

    return model

if __name__ == "__main__":
    try:
        model = load_model()
        predict_custom_test(model, device)
    except FileNotFoundError as e:
        print(e)
        exit(1)
