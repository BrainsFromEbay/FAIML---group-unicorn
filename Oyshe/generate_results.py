
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Constants
MODEL_PATH = "Oyshe/hog_logistic_mnist.npz"
TEST_DIR = "custom_test"
RESULTS_DIR = "Oyshe/results/hog_logreg"

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Model
try:
    data = np.load(MODEL_PATH)
    weights = data["weights"]
    bias = data["bias"]
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def preprocess_image(image_path):
    # Same preprocessing as predict.py
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    if np.mean(img) > 127:
        img = ImageOps.invert(img)

    img = np.array(img) / 255.0

    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(7, 7),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return hog_feat.reshape(1, -1)

def predict_digit(image_path):
    x = preprocess_image(image_path)
    z = np.dot(x, weights.T) + bias
    probs = softmax(z[0])
    digit = np.argmax(probs)
    return digit

def main():
    print(f"Evaluating model on {TEST_DIR}...")
    
    y_true = []
    y_pred = [] 
    
    files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith('.png')])
    
    for f in files:
        # Parse label: 0.png or 0(1).png -> 0
        try:
            label = int(f.split('(')[0].split('.')[0])
        except ValueError:
            print(f"Skipping bad filename: {f}")
            continue
            
        path = os.path.join(TEST_DIR, f)
        
        try:
            pred = predict_digit(path)
            y_true.append(label)
            y_pred.append(pred)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Generate Stats
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    
    print(f"Accuracy: {acc*100:.2f}%")
    
    stat_path = os.path.join(RESULTS_DIR, "stats.md")
    with open(stat_path, "w") as f:
        f.write("# Evaluation Results for HOG + Logistic Regression\n\n")
        f.write(f"**Accuracy**: {acc*100:.2f}%\n\n")
        f.write("## Classification Report\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
        
    print(f"Saved stats to {stat_path}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - HOG + Logistic Regression')
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()
