import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_images(file_path):
    with open(file_path, "rb") as f:
        f.read(4)  # magic
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)
    return images


def load_labels(file_path):
    with open(file_path, "rb") as f:
        f.read(4)
        num = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

X_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")

X_test = load_images("t10k-images.idx3-ubyte")
y_test = load_labels("t10k-labels.idx1-ubyte")

X_train = X_train / 255.0
X_test = X_test / 255.0

def extract_hog(images):
    features = []
    for img in images:
        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        features.append(hog_feat)
    return np.array(features)


print("Extracting HOG features...")
X_train_hog = extract_hog(X_train)
X_test_hog = extract_hog(X_test)

print("HOG feature shape:", X_train_hog.shape)

model = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    solver="lbfgs",
    n_jobs=-1
)

print("Training model...")
model.fit(X_train_hog, y_train)

y_pred = model.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

np.savez(
    "hog_logistic_mnist.npz",
    weights=model.coef_,
    bias=model.intercept_
)

print("Model saved as hog_logistic_mnist.npz")