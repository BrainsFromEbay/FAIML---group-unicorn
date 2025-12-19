import joblib
import numpy as np
import cv2
import os
model_path = 'Mahi/src/models/rf.joblib'
model = joblib.load(model_path)

folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Model loaded.")
print("-" * 55)
print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
print("-" * 55)

correct_count = 0
total_count = 0

for f in sorted(files):
    name = f.split('.')[0].strip()

    img_path = os.path.join(folder, f)
    
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        continue
    
    img_np = cv2.resize(img_np, (32, 32))

    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    img_flat = img_np.reshape(1, -1)
    
    input_data = img_flat.astype(np.float32) / 255.0

    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    confidence = probs[prediction] * 100

    print(f"{f:<20} | {prediction:<10} | {confidence:.1f}%")

    # Calculate accuracy
    try:
        base_name = f.split('(')[0]
        expected = int(base_name.split('.')[0])
        
        if expected == prediction:
            correct_count += 1
        total_count += 1
    except ValueError:
        pass

if total_count > 0:
    accuracy = (correct_count / total_count) * 100
    print("-" * 55)
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
    print("-" * 55)

