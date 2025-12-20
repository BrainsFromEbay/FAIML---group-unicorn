# Model Structure Report

This document details the specific "brain structures" (architectures) we built in our code. We have several variations of CNNs and MLPs, plus a Random Forest model.

---

## 1. Convolutional Neural Networks (CNN)

### A. `FinalCNN` Model
**File:** `src/train_script/CNN_mnist.py`
**Purpose:** Trained on standard MNIST dataset.

*   **Input:** 28x28 Grayscale Image (1 Channel)
*   **Structure:**
    1.  **Conv Layer 1**:
        *   Learns 16 filters (3x3).
        *   Preserves size (Padding=1).
        *   Activation: ReLU.
        *   **Pooling**: MaxPool (2x2) -> Image extracts become 14x14.
    2.  **Conv Layer 2**:
        *   Learns 32 filters (3x3) from the previous 16.
        *   Preserves size (Padding=1).
        *   Activation: ReLU.
        *   **Pooling**: MaxPool (2x2) -> Image extracts become 7x7.
    3.  **Classifier (Fully Connected)**:
        *   **Flatten**: Turns the 32 layers of 7x7 data into a list of 1,568 numbers.
        *   **Linear Layer**: Compresses to 128 features.
        *   **Dropout (25%)**: Randomly ignores 25% of neurons during training to prevent memorization.
        *   **Output Layer**: 10 neurons (Digits 0-9).

### B. `SimpleDigitCNN` Model
**File:** `src/train_script/CNN_pickle.py`
**Purpose:** Trained on our custom preprocessed pickled dataset.

*   **Start**: Slightly deeper than `FinalCNN`.
*   **Structure:**
    1.  **Block 1**:
        *   Conv (32 filters) -> ReLU -> Conv (32 filters) -> ReLU.
        *   **Pooling**: MaxPool (2x2).
    2.  **Block 2**:
        *   Conv (64 filters) -> ReLU -> Conv (64 filters) -> ReLU.
        *   **Pooling**: MaxPool (2x2).
    3.  **Classifier**:
        *   **Dropout (50%)**: Heavy regularization.
        *   **Linear Layer**: Compresses to 128 features.
        *   **Dropout (30%)**: More regularization.
        *   **Output Layer**: 10 neurons.

### C. `SimpleCNN` Model
**File:** `src/train_script/CNN_raw.py`
**Purpose:** Trained on raw image folders.

*   **Structure**: Similar to `FinalCNN` but with different filter counts.
    *   Conv (16 filters) -> ReLU -> Pool.
    *   Conv (32 filters) -> ReLU -> Pool.
    *   Linear (128 features) -> Output (10 classes).

---

## 2. Multi-Layer Perceptrons (MLP)

### A. `MNIST_MLP` Model
**File:** `src/train_script/MLP_mnist.py`
**Purpose:** Basic deep feed-forward network for MNIST.

*   **Input**: Flattened 28x28 image = 784 numbers.
*   **Structure**:
    1.  **Hidden Layer 1**: 512 Neurons (ReLU + Dropout 20%).
    2.  **Hidden Layer 2**: 256 Neurons (ReLU + Dropout 20%).
    3.  **Output Layer**: 10 Neurons.

### B. `SimpleMLP` Model
**File:** `src/train_script/MLP_pickle.py`
**Purpose:** Deeper MLP for custom data with Batch Normalization.

*   **Input**: Flattened 32x32 image (resized in code) = 1,024 numbers.
*   **Structure**:
    1.  **Layer 1**: 256 Neurons + **Batch Norm** + ReLU + Dropout (40%).
        *   *Batch Norm helps the model learn faster and more strictly.*
    2.  **Layer 2**: 128 Neurons + Batch Norm + ReLU + Dropout (40%).
    3.  **Layer 3**: 64 Neurons + Batch Norm + ReLU + Dropout (40%).
    4.  **Output**: 10 Neurons.

---

## 3. Random Forest

**Files:** `random_forrest.py` & `random_forrest_mnist.py`

*   **Structure**: It is not a neural network, so it has no "layers."
*   **Configuration**:
    *   **Size**: 100 Trees (`n_estimators=100`).
    *   **Jobs**: `-1` (Uses all CPU cores available for speed).
    *   **Input**: Flattened images (just like MLP).
