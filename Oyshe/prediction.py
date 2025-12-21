import os
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


def load_model(base_path="Oyshe"):
    npz_path = os.path.join(base_path, "hog_logistic_mnist.npz")
    if not os.path.exists(npz_path):
        # Fallback for when running directly
        npz_path = "hog_logistic_mnist.npz"
        
    data = np.load(npz_path)
    weights = data["weights"]
    bias = data["bias"]
    return weights, bias

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def preprocess_image(image_path):
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

def predict_digit(image_path, weights, bias):
    x = preprocess_image(image_path)
    z = np.dot(x, weights.T) + bias
    probs = softmax(z[0])

    digit = np.argmax(probs)
    confidence = probs[digit]

    return digit, confidence



from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def generate_results(weights, bias):
    # Determine base output path
    if os.path.basename(os.getcwd()) == "Oyshe":
        results_dir = "results"
        test_dir = "../custom_test"
    else:
        results_dir = "Oyshe/results"
        test_dir = "custom_test"
        
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
         # Try local fallback
         if os.path.exists("custom_test"):
             test_dir = "custom_test"
         else:
             print(f"Test dir not found: {test_dir}")
             return

    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    y_true = []
    y_pred = []
    
    print("-" * 50)
    print(f"{'File':<20} | {'True':<5} | {'Pred':<5} | {'Conf':<6}")
    print("-" * 50)

    for path in image_paths:
        filename = os.path.basename(path)
        try:
             # Assumes filename starts with digit like 0.png or 0(1).png
             label = int(filename[0])
        except:
             continue
             
        pred, conf = predict_digit(path, weights, bias)
        
        y_true.append(label)
        y_pred.append(pred)
        
        print(f"{filename:<20} | {label:<5} | {pred:<5} | {conf:.2f}")

    if not y_true:
        print("No valid images found.")
        return

    acc = accuracy_score(y_true, y_pred)
    print("-" * 50)
    print(f"Accuracy: {acc*100:.2f}%")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Oyshe')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion Matrix saved to {os.path.join(results_dir, 'confusion_matrix.png')}")


if __name__ == "__main__":
    # Determine base path based on where the script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        weights, bias = load_model(base_path)
        generate_results(weights, bias)
    except Exception as e:
        print(f"Error: {e}")


