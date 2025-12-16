import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------
# GPU Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------
# 1. Only Digits 0-9 (10 classes)
# -----------------------------
class_names = [str(i) for i in range(10)]  # '0','1',...,'9'
num_classes = 10
print(f"Training on digits only: {class_names}")

# -----------------------------
# 2. Custom Dataset (only loads digit folders)
# -----------------------------
class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in class_to_idx:
                print(f"Skipping folder: {class_name}")
                continue
                
            label = class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
        image = Image.fromarray(image)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# -----------------------------
# 3. Transforms: Resize + Flatten later in model
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),   # All images become 28x28
    transforms.ToTensor(),         # Converts to [1, 28, 28] tensor with values 0-1
])

train_dataset = DigitDataset(root_dir='Train', transform=transform)
val_dataset   = DigitDataset(root_dir='Validation', transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -----------------------------
# 4. Simple MLP (Fully Connected Neural Network)
# -----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),              # Turn 1x28x28 image into 784 vector
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)  # Output 10 scores (one per digit)
        )
    
    def forward(self, x):
        return self.network(x)

# Create model and send to GPU
model = SimpleMLP().to(device)
print(model)

# Count parameters (should be small!)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# -----------------------------
# 5. Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6. Training Loop
# -----------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                  # Compute gradients
        optimizer.step()                 # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), correct / total

def validate():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# -----------------------------
# 7. Train!
# -----------------------------
num_epochs = 15

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    val_acc = validate()
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Acc : {val_acc:.4f}")
    print("-" * 50)

torch.save(model.state_dict(), 'digits_model_MLP_from_raw_image.pth')
print("MLP model saved!")

# -----------------------------
# 8. Predict a single image
# -----------------------------
def predict_digit(image_path, model, transform):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dim
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Example after training:
# print("Predicted digit:", predict_digit('Train/5/some_image.png', model, transform))