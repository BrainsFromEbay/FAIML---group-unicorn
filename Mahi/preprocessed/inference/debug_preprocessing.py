import cv2
import os
import numpy as np

TEST_DIR = "custom_test"
OUTPUT_DIR = "Mahi/preprocessed/inference/debug_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"Processing {len(files)} images for debugging...")
print(f"Saving outputs to {OUTPUT_DIR}")

for filename in files:
    filepath = os.path.join(TEST_DIR, filename)
    
    # 1. Read
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # 2. Invert
    img_inv = 255 - img
    
    # 3. Resize
    img_resized = cv2.resize(img_inv, (32, 32))
    
    # 4. Threshold (The step we suspect causes issues)
    img_thresh_50 = img_resized.copy()
    img_thresh_50[img_thresh_50 < 50] = 0
    
    # 5. Threshold (Alternative: Is 20 better?)
    img_thresh_20 = img_resized.copy()
    img_thresh_20[img_thresh_20 < 20] = 0
    
    # Save comparison
    # Stack images horizontally: Original(Resized+Inv) | Thresh 50 | Thresh 20
    combined = np.hstack((img_resized, img_thresh_50, img_thresh_20))
    
    save_path = os.path.join(OUTPUT_DIR, f"debug_{filename}")
    cv2.imwrite(save_path, combined)
    print(f"Saved {save_path}")

print("Debug images created. Open the folder to inspect.")
