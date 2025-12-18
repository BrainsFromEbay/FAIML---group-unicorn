import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast as autocast_cuda
import torchvision.transforms as transforms
import sys

# CONFIGURATION
PICKLE_FILE = "Mahi/preprocessed/digits_data.pickle" # Raw data
BATCH_SIZE = 1024 # Increased from 64 for speed
NUM_EPOCHS = 10 # 10 epochs is likely sufficient given fast convergence
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DigitsDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        
        # Handle shapes
        if len(image.shape) == 3 and image.shape[2] == 1:
             image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # (1, 32, 32)
        elif len(image.shape) == 2:
             image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
             image_tensor = torch.tensor(image, dtype=torch.float32)

        # Scale to [0, 1] first
        image_tensor = image_tensor / 255.0

        # Apply transforms (Augmentation + Normalization)
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, self.y[idx]

# Friend's Architecture (Adapted for 32x32)
class InspiredCNN(nn.Module):
    def __init__(self):
        super(InspiredCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        # Friend had 28x28 -> 7x7
        self.fc1 = nn.Linear(32 * 8 * 8, 128) 
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

def load_data():
    print(f"Loading data from {PICKLE_FILE}...", flush=True)
    with open(PICKLE_FILE, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully.", flush=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val']

if __name__ == '__main__':
    print(f"Using device: {DEVICE}", flush=True)
    
    X_train, y_train, X_val, y_val = load_data()

    # Friend's Augmentation + Normalization
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        # Normalize to [-1, 1] (Assuming data is [0, 1] entering transform)
        # mean=0.5, std=0.5 -> (x - 0.5) / 0.5
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = DigitsDataset(X_train, y_train, transform=train_transform)
    val_dataset = DigitsDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = InspiredCNN().to(DEVICE)
    print(model, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    best_val_acc = 0.0

    print("Starting training...", flush=True)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            if DEVICE.type == 'cuda':
                with autocast_cuda():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images) # No autocast for val usually fine, or keep it consistent
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Mahi/preprocessed/models/best_cnn_inspired.pth")

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
