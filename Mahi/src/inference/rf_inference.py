import joblib
import numpy as np
import cv2
import os
model_path = 'Mahi/src/models/rf.joblib'
model = joblib.load(model_path)

folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Found {len(files)} images in '{folder}'.\n")
print(f"Using model: {model_path}")
print("-" * 40)

for f in sorted(files):
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    img_path = os.path.join(folder, f)
    
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    img_np = cv2.resize(img_np, (32, 32))

    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    img_flat = img_np.reshape(1, -1)
    
    input_data = img_flat.astype(np.float32) / 255.0

    prediction = model.predict(input_data)[0]

    print(f"File: {f}")
    print(f"Expected digit (from filename): {expected}")
    print(f"Predicted digit: {prediction}")
    print(f"Match: {'Yes' if expected == prediction else 'No'}")
    print("-" * 40)

print("\nInference complete.")
