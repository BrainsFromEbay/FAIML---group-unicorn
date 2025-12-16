import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

# ==================== Configuration ====================
PICKLE_FILE = "Mahi/preprocessed/digits_data_cleaned.pickle"
MODEL_FILE = "Mahi/preprocessed/rf_model.joblib"

# ==================== Load Data ====================
print("Loading cleaned dataset...")
with open(PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']  # (416126, 32, 32, 1)
y_train = data['y_train']
X_val = data['X_val']      # (11453, 32, 32, 1)
y_val = data['y_val']

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

# ==================== Preprocess: Flatten and Normalize ====================
print("Flattening and normalizing data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
X_val_flat = X_val.reshape(X_val.shape[0], -1).astype(np.float32) / 255.0

# ==================== Initialize and Train Model ====================
print("Initializing Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all CPU cores

start_time = time.time()
print("Training...")
model.fit(X_train_flat, y_train)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

# ==================== Evaluate on Validation Set ====================
print("Evaluating...")
y_pred = model.predict(X_val_flat)

accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ==================== Save Model ====================
print(f"Saving model to '{MODEL_FILE}'...")
joblib.dump(model, MODEL_FILE)
print("Done! Use joblib.load() for inference.")