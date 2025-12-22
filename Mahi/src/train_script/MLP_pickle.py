import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast as autocast_cuda
import torchvision.transforms as transforms
from PIL import Image
import sys

PICKLE_FILE = "Mahi/src/data_preprocessing/digits_data_cleaned.pickle"
BATCH_SIZE = 256
NUM_EPOCHS = 100 
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10 

class DigitsDatasetMLP(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        if len(image.shape) == 1:
            image = image.reshape(32, 32)
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        image_flat = image_tensor.view(-1) / 255.0
        
        label = self.y[idx]
        return image_flat, label

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

def load_data():
    print(f"Loading data from {PICKLE_FILE}...", flush=True)
    with open(PICKLE_FILE, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully.", flush=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val']

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if DEVICE.type == 'cuda':
                with autocast_cuda():
                    outputs = model(images)
            else:
                outputs = model(images)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

if __name__ == '__main__':
    print(f"Using device: {DEVICE}", flush=True)
    X_train, y_train, X_val, y_val = load_data()
    print(f"Training samples: {len(y_train)}", flush=True)
    print(f"Validation samples: {len(y_val)}", flush=True)

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    train_dataset = DigitsDatasetMLP(X_train, y_train, transform=train_transform)
    val_dataset = DigitsDatasetMLP(X_val, y_val, transform=None)

    pin_mem = (DEVICE.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model = SimpleMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()

    print(model, flush=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}", flush=True)

    start_time = time.time()
    best_val_acc = 0.0
    counter = 0 
    
    print("Starting training...", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_acc = validate(model, val_loader)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "Mahi/src/models/best_mlp_improved.pth")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.", flush=True)
                break

    total_time = time.time() - start_time
    print(f"\nMLP Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", flush=True)

    torch.save(model, "Mahi/src/models/MLP_pickle.pth")
