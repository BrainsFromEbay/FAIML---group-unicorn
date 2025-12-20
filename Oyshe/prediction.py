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


if __name__ == "__main__":
    # Determine base path based on where the script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    weights, bias = load_model(base_path)

    folder = "custom_test"
    
    print("Predicting hand-written digits:\n")
    
    if os.path.exists(folder):
        for file in sorted(os.listdir(folder)):
            if file.endswith(".png"):
                path = os.path.join(folder, file)
                digit, conf = predict_digit(path, weights, bias)
                print(f"{file}  -->  Digit: {digit}  |  Confidence: {conf:.2f}")

