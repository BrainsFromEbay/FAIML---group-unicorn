import joblib
import numpy as np
import os
from PIL import Image
MODEL_PATH = "Mahi/src/models/rf_mnist.joblib"
TEST_DIR = "custom_test"

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
            
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        
        img_flat = img_arr.reshape(1, -1)
        return img_flat
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28), Image.BILINEAR)
        img_arr = np.array(image)
        
        if img_arr.mean() > 127:
            img_arr = 255 - img_arr
            
        img_arr = img_arr.astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        
        img_flat = img_arr.reshape(1, -1)
        return img_flat
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    print(f"Loading RF Model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print("-" * 55)
    print(f"{'Filename':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 55)

    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in files:
        filepath = os.path.join(TEST_DIR, filename)
        input_flat = preprocess_image(filepath)
        
        if input_flat is None:
            continue
        
        pred = model.predict(input_flat)[0]
        
        probs = model.predict_proba(input_flat)[0]
        confidence = probs[pred] * 100
        
        print(f"{filename:<20} | {pred:<10} | {confidence:.1f}%")

if __name__ == "__main__":
    main()
