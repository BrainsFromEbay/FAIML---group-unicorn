# Oyshe

## Dataset

The dataset used for training is the **MNIST** dataset (28x28 grayscale images of handwritten digits).
The project includes the raw MNIST byte files:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Setup and Installation

This directory uses standard Python libraries for image processing and machine learning.

**Dependencies:**
- `numpy`
- `pillow` (PIL)
- `scikit-image` (skimage) used for HOG feature extraction.
- `scikit-learn` (implicitly used for Logistic Regression training, though model is saved as weights).

To run the prediction script:
```bash
python Oyshe/prediction.py
```

---

## Project Workflows

This folder implements a **HOG (Histogram of Oriented Gradients) + Logistic Regression** pipeline.

1.  **Preprocessing**:
    - Images are resized to 28x28.
    - Inverted if the background is light (standard MNIST is white-on-black).
    - HOG features are extracted using `skimage.feature.hog` (9 orientations, 7x7 cells, 2x2 blocks).

2.  **Model**:
    - A Logistic Regression model (linear classifier).
    - Trained weights and bias are saved in `hog_logistic_mnist.npz`.

3.  **Inference**:
    - `prediction.py`: Loads the weights, computes HOG features for custom images, and predicts the digit.

---

## File Descriptions

- `README.md`: This file.
- `model.py`: Script used to train the model (presumably, based on artifacts).
- `prediction.py`: Script to run inference on new images.
- `hog_logistic_mnist.npz`: Saved model weights and biases.

---

# Model Performance Comparison Report

The model was evaluated on the `custom_test` dataset (20 images).

## Prediction Matrix (Full Test Set)

| Image | Expected | Predicted | Confidence | Result |
| :---: | :------: | :-------: | :--------: | :----: |
| 0(1).png | 0 | 0 | 100.0% | ✅ |
| 0.png    | 0 | 0 | 96.0%  | ✅ |
| 1(1).png | 1 | 1 | 100.0% | ✅ |
| 1.png    | 1 | 1 | 100.0% | ✅ |
| 2(1).png | 2 | 2 | 73.0%  | ✅ |
| 2.png    | 2 | 2 | 93.0%  | ✅ |
| 3(1).png | 3 | 3 | 100.0% | ✅ |
| 3.png    | 3 | 9 | 80.0%  | ❌ |
| 4(1).png | 4 | 7 | 92.0%  | ❌ |
| 4.png    | 4 | 5 | 42.0%  | ❌ |
| 5(1).png | 5 | 5 | 100.0% | ✅ |
| 5.png    | 5 | 5 | 78.0%  | ✅ |
| 6(1).png | 6 | 5 | 76.0%  | ❌ |
| 6.png    | 6 | 9 | 71.0%  | ❌ |
| 7(1).png | 7 | 7 | 92.0%  | ✅ |
| 7.png    | 7 | 7 | 95.0%  | ✅ |
| 8(1).png | 8 | 8 | 84.0%  | ✅ |
| 8.png    | 8 | 9 | 47.0%  | ❌ |
| 9(1).png | 9 | 3 | 94.0%  | ❌ |
| 9.png    | 9 | 1 | 73.0%  | ❌ |

## Accuracy Scoreboard

| Metric | Score |
| :--- | :---: |
| **Correct** | **12 / 20** |
| **Accuracy** | **60.0%** |

## Analysis

1.  **Performance**: The HOG + Logistic Regression model achieved **60% accuracy** overall.
2.  **Noisy vs Clean**: Interestingly, the model performed **better** on some of the "noisy" `(1).png` images than the clean ones.
    - **Correctly identified**: `3(1)` and `8(1)` were classified correctly, while their clean counterparts `3` and `8` were misclassified as `9`.
    - This suggests the HOG features (gradients) might be more distinct or robust in the specific "noisy" style for these digits, or the noise ironically helps bridge disconnected strokes that HOG might otherwise misinterpret.
3.  **Use Cases**: This model is surprisingly robust for digits 0, 1, 2, 5, and 7, achieving 100% accuracy on those pairs. It fails consistently on 4, 6, and 9.

## Conclusion

HOG + Logistic Regression provides a decent baseline (60%), performing better than expected and even outperforming some Neural Networks (like `CNN Pickle` at 25%) on this specific dataset. It shows that feature engineering can still be effective.
