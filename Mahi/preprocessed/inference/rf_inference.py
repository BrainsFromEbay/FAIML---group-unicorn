import joblib
import numpy as np
import cv2
import os

# 1. Load the trained Random Forest model
model_path = 'Mahi/preprocessed/models/rf.joblib'
model = joblib.load(model_path)

# 2. Preprocessing and Prediction Loop
folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Found {len(files)} images in '{folder}'.\n")
print(f"Using model: {model_path}")
print("-" * 40)

for f in sorted(files):
    # Get the expected digit from the filename
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    # --- Image Preprocessing ---
    # This pipeline matches the one used for PyTorch models trained on the pickled data,
    # and also ensures the data is flattened and scaled as expected by the Random Forest model.
    img_path = os.path.join(folder, f)
    
    # Read in grayscale
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 32x32, which is the input size for the pickled data
    img_np = cv2.resize(img_np, (32, 32))

    # Invert colors (black background, white digit)
    img_np = 255 - img_np

    # Binarize the image to remove noise
    img_np[img_np < 50] = 0
    
    # Flatten the image to a 1D array (32*32=1024 features)
    # The model expects a 2D array: (1, 1024) for a single image
    img_flat = img_np.reshape(1, -1)
    
    # Scale pixel values to [0.0, 1.0]
    input_data = img_flat.astype(np.float32) / 255.0

    # --- Prediction ---
    prediction = model.predict(input_data)[0]

    print(f"File: {f}")
    print(f"Expected digit (from filename): {expected}")
    print(f"Predicted digit: {prediction}")
    print(f"Match: {'Yes' if expected == prediction else 'No'}")
    print("-" * 40)

print("\nInference complete.")
