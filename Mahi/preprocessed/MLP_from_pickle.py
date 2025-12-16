import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast

# ==================== Configuration ====================
PICKLE_FILE = "Mahi/preprocessed/digits_data_cleaned.pickle"
BATCH_SIZE = 256                  # MLP is lighter, so we can use larger batch
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== Dataset (flatten images) ====================
class DigitsDatasetMLP(Dataset):
    def __init__(self, X, y):
        self.X = X  # Keep as uint8 to save memory
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]  # (32, 32, 1)
        image = image.reshape(-1)  # Flatten to 1024 values
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize
        
        label = self.y[idx]
        return image, label

# ==================== Simple MLP Model ====================
class SimpleMLP(nn.Module):
    def __init__(self, input_size=32*32, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ==================== Load Data ====================
print("Loading cleaned dataset...")
with open(PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']  # (N, 32, 32, 1)
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

train_dataset = DigitsDatasetMLP(X_train, y_train)
val_dataset = DigitsDatasetMLP(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ==================== Model, Loss, Optimizer ====================
model = SimpleMLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # Mixed precision for speed & low VRAM

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# ==================== Training Functions ====================
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

# ==================== Start Training ====================
if __name__ == '__main__':
    print("\nStarting MLP training (no CNN)...\n")
    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch()
        val_acc = validate()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Mahi/preprocessed/best_mlp_model.pth")
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\nMLP Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Best MLP weights saved as 'best_mlp_model.pth'")

    # Save full model for easy inference
    torch.save(model, "Mahi/preprocessed/mlp_full.pth")
    print("Full MLP model saved as 'mlp_full.pth'")