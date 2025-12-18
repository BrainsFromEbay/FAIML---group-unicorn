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
PICKLE_FILE = "Mahi/preprocessed/digits_data.pickle" # Using RAW data this time
BATCH_SIZE = 512 # Increased for GPU utilization
NUM_EPOCHS = 5 # Reduced from 50 as per user request (convergence seen early)
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10 

# Enable CuDNN benchmark for speed
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

class DigitsDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        
        # Ensure (H, W, C) -> (C, H, W) or (H, W) -> (1, H, W)
        # Based on previous scripts, input might be (32, 32, 1) or flat
        if len(image.shape) == 1:
             # If flat, reshape. Assuming 32x32 based on project context
             # But let's assume valid input shape from raw pickle
             pass
        
        # Convert to tensor
        # Raw data might be uint8 0-255
        if len(image.shape) == 3 and image.shape[2] == 1:
             image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        elif len(image.shape) == 2:
             image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
             image_tensor = torch.tensor(image, dtype=torch.float32)

        # Apply transforms if present
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        # Normalize to 0-1
        return image_tensor / 255.0, self.y[idx]

class RefinedDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RefinedDigitCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16 -> 8x8
        )
        
        # Reverting to Dense Layers for specific spatial feature retention
        # But keeping it lighter than the original 400k+ params
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), # 4096 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Stronger dropout for dense layer
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
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
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
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

    # Mild Augmentation (Refined)
    train_transform = transforms.Compose([
        transforms.RandomRotation(5), # +/- 5 degrees (was 10)
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # +/- 5% shift (was 10%)
    ])

    train_dataset = DigitsDataset(X_train, y_train, transform=train_transform)
    val_dataset = DigitsDataset(X_val, y_val, transform=None)

    pin_mem = (DEVICE.type == 'cuda')
    num_workers = 4 # Increased for speed
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, persistent_workers=True)

    model = RefinedDigitCNN(num_classes=10).to(DEVICE)
    print(model, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Keeping regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler()

    start_time = time.time()
    best_val_acc = 0.0
    counter = 0

    print("Starting training...", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_acc = validate(model, val_loader)
        epoch_dur = time.time() - epoch_start
        
        # Step scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:02d} | Time: {epoch_dur:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "Mahi/preprocessed/models/best_cnn_refined.pth")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.", flush=True)
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", flush=True)

    torch.save(model, "Mahi/preprocessed/models/cnn_refined_full.pth")
