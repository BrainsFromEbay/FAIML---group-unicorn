
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "custom_test"
RESULTS_BASE = "Mahi/results"

# Ensure results directory exists
os.makedirs(RESULTS_BASE, exist_ok=True)

# --- Model Definitions ---

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

class SimpleDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FinalCNN_MNIST(nn.Module):
    def __init__(self):
        super(FinalCNN_MNIST, self).__init__()
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

# --- Preprocessing Functions ---

def preprocess_raw(path):
    # Resize 28x28, Invert, Threshold<50
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_np is None: return None
    img_np = 255 - img_np
    img_np[img_np < 50] = 0
    image_pil = Image.fromarray(img_np)
    img_tensor = transform(image_pil)
    return img_tensor.unsqueeze(0).to(DEVICE)

def preprocess_pickle_cnn(path):
    # Resize 32x32, Invert, Threshold<50, /255.0
    img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_np is None: return None
    img_np = cv2.resize(img_np, (32, 32))
    img_np = 255 - img_np
    img_np[img_np < 50] = 0
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32)
    return img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

def preprocess_pickle_mlp(path):
    # Resize 32x32, Invert, Threshold<50, Flatten, /255.0
    img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_np is None: return None
    img_np = 255 - img_np
    img_np = cv2.resize(img_np, (32, 32))
    img_np[img_np < 50] = 0
    img_flat = img_np.reshape(-1)
    img_normalized = img_flat.astype(np.float32) / 255.0
    return torch.tensor(img_normalized).unsqueeze(0).to(DEVICE)

def preprocess_mnist_cnn(path):
    # Resize 28x28, Invert if mean>127, (x-0.5)/0.5
    try:
        image = Image.open(path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)
        return img_tensor.to(DEVICE)
    except:
        return None

def preprocess_mnist_mlp(path):
    # Same as CNN MNIST but potentially flat handled by model, wait model has flatten.
    # Yes, model has flatten. Preprocessing returns (1,1,28,28).
    return preprocess_mnist_cnn(path)

# RF Preprocessors follow similar logic to MLPs
def preprocess_rf_pickle(path):
    img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_np is None: return None
    img_np = cv2.resize(img_np, (32, 32))
    img_np = 255 - img_np
    img_np[img_np < 50] = 0
    img_flat = img_np.reshape(1, -1)
    return img_flat.astype(np.float32) / 255.0

def preprocess_rf_mnist(path):
    try:
        image = Image.open(path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_flat = img_arr.reshape(1, -1)
        return img_flat
    except:
        return None

# --- Main Evaluation Loop ---

def evaluate_model(name, model, preprocess_func, is_torch=True):
    print(f"Evaluating {name}...")
    y_true = []
    y_pred = []
    
    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith('.png')])
    for f in files:
        # Parse label: 0.png or 0(1).png -> 0
        try:
            label = int(f.split('(')[0].split('.')[0])
        except:
            continue
            
        path = os.path.join(TEST_DIR, f)
        input_data = preprocess_func(path)
        if input_data is None: continue
        
        y_true.append(label)
        
        if is_torch:
            with torch.no_grad():
                output = model(input_data)
                pred = output.argmax(dim=1).item()
                y_pred.append(pred)
        else:
            # sklearn model
            pred = model.predict(input_data)[0]
            try:
                pred = int(pred) # ensure int
            except:
                pass
            y_pred.append(pred)

    # Generate Results
    out_dir = os.path.join(RESULTS_BASE, name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Stats
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    
    with open(os.path.join(out_dir, "stats.md"), "w") as f:
        f.write(f"# Evaluation Results for {name}\n\n")
        f.write(f"**Accuracy**: {acc*100:.2f}%\n\n")
        f.write("## Classification Report\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Saved results to {out_dir}")

def main():
    # 1. CNN Raw
    try:
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load('Mahi/src/models/CNN_raw.pth', map_location=DEVICE))
        model.eval()
        evaluate_model('cnn_raw', model, preprocess_raw) 
    except Exception as e:
        print(f"Skipping CNN Raw: {e}")

    # 2. CNN Pickle
    try:
        model = SimpleDigitCNN().to(DEVICE)
        # Load carefuly
        m_path = 'Mahi/src/models/CNN_pickle.pth'
        loaded = torch.load(m_path, map_location=DEVICE)
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        model.eval()
        evaluate_model('cnn_pickle', model, preprocess_pickle_cnn)
    except Exception as e:
        print(f"Skipping CNN Pickle: {e}")

    # 3. CNN MNIST
    try:
        model = FinalCNN_MNIST().to(DEVICE)
        model.load_state_dict(torch.load('Mahi/src/models/CNN_mnist.pth', map_location=DEVICE))
        model.eval()
        evaluate_model('cnn_mnist', model, preprocess_mnist_cnn)
    except Exception as e:
        print(f"Skipping CNN MNIST: {e}")

    # 4. MLP Pickle
    try:
        model = SimpleMLP().to(DEVICE)
        # Original script had issues loading state_dict, tried full model.
        m_path = 'Mahi/src/models/MLP_pickle.pth'
        try:
            model.load_state_dict(torch.load(m_path, map_location=DEVICE))
        except:
             model = torch.load(m_path, map_location=DEVICE)
        model.eval()
        evaluate_model('mlp_pickle', model, preprocess_pickle_mlp)
    except Exception as e:
        print(f"Skipping MLP Pickle: {e}")
        
    # 5. MLP MNIST
    try:
        model = MNIST_MLP().to(DEVICE)
        model.load_state_dict(torch.load('Mahi/src/models/MLP_mnist.pth', map_location=DEVICE))
        model.eval()
        evaluate_model('mlp_mnist', model, preprocess_mnist_mlp)
    except Exception as e:
        print(f"Skipping MLP MNIST: {e}")

    # 6. RF Pickle
    try:
        model = joblib.load('Mahi/src/models/rf.joblib')
        evaluate_model('rf_pickle', model, preprocess_rf_pickle, is_torch=False)
    except Exception as e:
        print(f"Skipping RF Pickle: {e}")
        
    # 7. RF MNIST
    try:
        model = joblib.load('Mahi/src/models/rf_mnist.joblib')
        evaluate_model('rf_mnist', model, preprocess_rf_mnist, is_torch=False)
    except Exception as e:
        print(f"Skipping RF MNIST: {e}")

if __name__ == "__main__":
    main()
