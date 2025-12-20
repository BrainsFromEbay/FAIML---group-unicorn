# Data Preprocessing Documentation

This document details the data preprocessing pipelines used in the project. The project utilizes two primary data sources: the standard MNIST dataset and a custom dataset of handwritten digits.

## 1. Offline Preprocessing (Custom Data)

The custom dataset undergoes a two-step "offline" preprocessing phase before being used for training. These scripts are located in `Mahi/src/data_preprocessing`.

### Step 1: Raw Image Conversion
**Script:** `raw_image_to_binary.py`

This script converts raw image files into a structured pickle format.
*   **Input:** Images from `Train` and `Validation` directories, organized by class folders ('0'-'9').
*   **Process:**
    1.  Reads images in Grayscale mode (`cv2.IMREAD_GRAYSCALE`).
    2.  Resizes images to **32x32** pixels.
    3.  Reshapes the data into a 4D tensor with the format `(N, 32, 32, 1)` to prepare it for deep learning models:
        *   **N**: The total number of images in the dataset.
        *   **32, 32**: The height and width of each image in pixels.
        *   **1**: The number of color channels (1 for grayscale).
*   **Output:** `digits_data.pickle`

### Step 2: Cleaning and Denoising
**Script:** `pickle_clean.py`

This script cleans the data generated in Step 1 to remove noise.
*   **Input:** `digits_data.pickle`
*   **Process:**
    *   Applies a **thresholding** operation: Any pixel value less than **50** is set to **0**. This helps remove background noise and faint artifacts.
*   **Output:** `digits_data_cleaned.pickle` (Used by `*_pickle.py` training scripts).

---

## 2. Online Preprocessing (Training Scripts)

Each training script applies specific transformations during the data loading phase.

### A. MNIST Based Scripts
These scripts download and process the standard MNIST dataset (28x28 images).

| Script | Model Type | Preprocessing Steps |
| :--- | :--- | :--- |
| `CNN_mnist.py` | CNN | 1. **Resize:** 28x28<br>2. **Augmentation:** RandomAffine (deg=15, trans=0.1, scale=0.9-1.1)<br>3. **Tensor:** Converts to [0, 1]<br>4. **Normalize:** `mean=(0.5,), std=(0.5,)` → Range **[-1, 1]** |
| `MLP_mnist.py` | MLP | Same as `CNN_mnist.py`. Flattening is handled inside the model's `forward` method. |
| `random_forrest_mnist.py` | Random Forest | 1. **Tensor:** [0, 1]<br>2. **Normalize:** `mean=(0.5,), std=(0.5,)` → [-1, 1]<br>3. **Numpy Convert:** Converts back to numpy.<br>4. **Flatten:** Reshapes to `(N, 784)`. |

### B. Custom Data (Pickle) Scripts
These scripts load the pre-processed `digits_data_cleaned.pickle` (32x32 images).

| Script | Model Type | Preprocessing Steps |
| :--- | :--- | :--- |
| `CNN_pickle.py` | CNN | 1. **Scale:** Divides by 255.0 → Range **[0, 1]**<br>2. **Permute:** Changes shape from `(32, 32, 1)` to `(1, 32, 32)` (CHW format). |
| `MLP_pickle.py` | MLP | 1. **Augmentation:** RandomRotation(10), RandomAffine(trans=0.1)<br>2. **Scale:** Divides by 255.0 → Range **[0, 1]**<br>3. **Flatten:** Reshapes to `(1024,)`. |
| `random_forrest.py` | Random Forest | 1. **Structure:** Reshapes to `(N, 1024)`.<br>2. **Scale:** Divides by 255.0 → Range **[0, 1]**. |

### C. Raw Image Script
This script loads images directly from folders without using the pickle file.

| Script | Model Type | Preprocessing Steps |
| :--- | :--- | :--- |
| `CNN_raw.py` | CNN | 1. **Grayscale:** Loads as grayscale.<br>2. **Resize:** 28x28.<br>3. **Tensor:** Converts to [0, 1]. **No Normalization** to [-1, 1]. |

## Summary of Input Shapes

*   **MNIST Models:** 28x28 input.
*   **Pickle Models:** 32x32 input.
*   **Raw Models:** 28x28 input.

> **Note:** There is a discrepancy in normalization. MNIST models are typically normalized to `[-1, 1]`, while Custom Data models are scaled to `[0, 1]`.
