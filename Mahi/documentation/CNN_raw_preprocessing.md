# CNN_raw.py Data Preprocessing Documentation

This document provides a detailed breakdown of how `Mahi/src/train_script/CNN_raw.py` handles data loading and preprocessing. Unlike the "pickle" based scripts, this script loads raw image files directly from the disk.

## Overview
The script uses a custom `Dataset` class named `CharacterDataset` to load images from a directory structure where subdirectories represent class labels.

**Source Code Reference:** `Mahi/src/train_script/CNN_raw.py`

## 1. Data Loading (`CharacterDataset`)

The `CharacterDataset` class is responsible for iterating through the filesystem and loading images.

### Directory Structure Assumption
The script expects the following root directory structure (configurable via arguments, defaults to `'Train'` and `'Validation'`):
```
Root_Dir/
  ├── 0/
  │   ├── image1.png
  │   └── ...
  ├── 1/
  └── ...
```

### Class Filtering
*   **Target Classes:** Defined explicitly as `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`.
*   **Sorting:** Class names are sorted alphabetically.
*   **Filtering:** Any subdirectory in the root folder that does not match a name in the target class list is skipped.

### Image Reading (`__getitem__`)
For each sample, the following steps occur:
1.  **Read File:** Uses OpenCV (`cv2.imread`) to read the image.
    *   **Flag:** `cv2.IMREAD_GRAYSCALE` - Loads the image in grayscale mode (1 channel).
2.  **Conversion to PIL Image (`Image.fromarray`)**:
    *   **Data Format Shift:** OpenCV reads images as **NumPy arrays**. However, the `torchvision.transforms` library is primarily designed to operate on **PIL Images**.
    *   **Bridge:** `Image.fromarray(image)` converts the NumPy array into a PIL Image object.
    *   **Necessity:** This conversion is required because subsequent steps like `transforms.Resize` will throw an error if passed a raw NumPy array instead of a PIL Image.

## 2. Transformation Pipeline

The script applies a sequence of transformations defined in `transforms.Compose`.

```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
```

### Step-by-Step Transformation
1.  **Resize (`transforms.Resize((28, 28))`)**
    *   **Action:** Resizes the input PIL Image to **28 pixels height x 28 pixels width**.
    *   **Interpolation (Bilinear):**
        *   **Concept:** When an image is resized, the new pixel grid does not perfectly align with the original. Interpolation is the mathematical process used to estimate the value of these "missing" pixels.
        *   **Mechanism:** Bilinear interpolation calculates the value of a new pixel by taking a weighted average of the **four nearest pixels** (2x2 neighborhood) from the original image.
        *   **Result:** This method provides a balance between computational speed and image quality, producing smoother results than "Nearest Neighbor" interpolation, which can leave images looking jagged or "pixelated."
    *   **Purpose:** Ensures all input images have the same dimensions required by the CNN architecture.

2.  **ToTensor (`transforms.ToTensor()`)**
    *   **Action:** Converts the PIL Image (values [0, 255]) to a PyTorch Tensor (values [0.0, 1.0]).
    *   **Shape Change:** Converts `(H, W, C)` (or `(H, W)` for grayscale) to `(C, H, W)`.
        *   Input: `(28, 28)` (Grayscale PIL)
        *   Output: `(1, 28, 28)` (Float Tensor)
    *   **Scaling:** Pixel values are scaled from the range `[0, 255]` to `[0.0, 1.0]`.

## 3. Difference from Other Scripts
*   **No Normalization:** Unlike `CNN_mnist.py` or the `MLP` scripts, simple normalization (e.g., `(image - 0.5) / 0.5`) is **NOT** applied here. The input values fed to the network remain in the `[0, 1]` range.
*   **No Augmentation:** There are no data augmentation steps (like rotation or affine transforms) applied in this pipeline.
