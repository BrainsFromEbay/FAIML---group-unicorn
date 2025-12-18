import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os

# CONFIGURATION
MODEL_SAVE_PATH = "Mahi/preprocessed/models/rf_mnist.joblib"

if __name__ == '__main__':
    # We load MNIST using PyTorch purely for convenience of downloading/loading
    # But we need numpy arrays for sklearn
    
    print("Loading MNIST Dataset...")
    try:
        # We don't need complex transforms for RF, just ToTensor matches 0-1 range.
        # However, to be consistent with our "Friend's Approach" that works well:
        # Friend used: Resize 28 (MNIST is 28), Invert if needed, Norm [-1, 1].
        # RFs don't care about Norm [-1, 1] vs [0, 1] usually, but let's stick to the 
        # exact same input distribution as the successful MLP/CNN to be safe.
        # Actually, sklearn RF might prefer 0-1 or 0-255. 
        # But if I use the SAME preprocessing as inference_mlp_mnist (which did [-1, 1]),
        # then I should train on [-1, 1].
        
        transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize((0.5,), (0.5,)) # [-1, 1]
        ])
        
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        
        # Convert to Numpy
        print("Converting to Numpy...")
        X_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        X_val = val_dataset.data.numpy()
        y_val = val_dataset.targets.numpy()
        
        # Wait, train_dataset.data is Raw uint8 (0-255). The transform is applied in __getitem__.
        # So I need to manually apply the transform or iterate.
        # Iterating is slow. Let's vectorise.
        
        # Raw MNIST is 0=Black, 255=White (Digits are white).
        # Our "Friend's Logic" in inference checked "if mean > 127 invert".
        # MNIST is consistent, so we don't need per-image invert check during training,
        # but we do need to match the normalization.
        
        # 1. Float & Scale to [0, 1]
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        
        # 2. Normalize to [-1, 1] ((x - 0.5) / 0.5)
        X_train = (X_train - 0.5) / 0.5
        X_val = (X_val - 0.5) / 0.5
        
        # 3. Flatten (N, 28*28)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        
        # Initialize and Train RF
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        print("Training Random Forest...")
        clf.fit(X_train_flat, y_train)
        
        print("Evaluating...")
        y_pred = clf.predict(X_val_flat)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc*100:.2f}%")
        
        # Save
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
