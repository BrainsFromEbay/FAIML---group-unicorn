import os
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

data = np.load("hog_logistic_mnist.npz")
weights = data["weights"]
bias = data["bias"]

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

def predict_digit(image_path):
    x = preprocess_image(image_path)
    z = np.dot(x, weights.T) + bias
    probs = softmax(z[0])

    digit = np.argmax(probs)
    confidence = probs[digit]

    return digit, confidence

folder = "Hand written number"

print("Predicting hand-written digits:\n")

for file in sorted(os.listdir(folder)):
    if file.endswith(".png"):
        path = os.path.join(folder, file)
        digit, conf = predict_digit(path)
        print(f"{file}  -->  Digit: {digit}  |  Confidence: {conf:.2f}")
