import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os

MODEL_SAVE_PATH = "Mahi/preprocessed/models/rf_mnist.joblib"

if __name__ == '__main__':
    
    print("Loading MNIST Dataset...")
    try:        
        transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize((0.5,), (0.5,)) # [-1, 1]
        ])
        
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        
        print("Converting to Numpy...")
        X_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        X_val = val_dataset.data.numpy()
        y_val = val_dataset.targets.numpy()
        
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        
        X_train = (X_train - 0.5) / 0.5
        X_val = (X_val - 0.5) / 0.5
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        print("Training Random Forest...")
        clf.fit(X_train_flat, y_train)
        
        print("Evaluating...")
        y_pred = clf.predict(X_val_flat)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc*100:.2f}%")
        
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
