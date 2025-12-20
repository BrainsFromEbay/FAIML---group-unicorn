# Preprocessing Comparison: CNN_pickle vs CNN_raw

This document compares the data preprocessing pipelines associated with two specific training scripts:
1.  `Mahi/src/train_script/CNN_pickle.py`
2.  `Mahi/src/train_script/CNN_raw.py`

These two scripts feed data into Convolutional Neural Networks (CNNs) but differ significantly in how they obtain and shape that data.

## At a Glance Comparison

| Feature | `CNN_pickle.py` | `CNN_raw.py` |
| :--- | :--- | :--- |
| **Data Source** | `digits_data_cleaned.pickle` (Pre-processed) | Raw Images from Folders (`Train`/`Validation`) |
| **Input Resolution** | **32 x 32** | **28 x 28** |
| **Data Cleaning** | **Yes** (Offline: pixels < 50 set to 0) | **No** (Direct resize of raw image) |
| **Normalization** | Scaled to **[0, 1]** (div by 255.0) | Scaled to **[0, 1]** (via `ToTensor`) |
| **Tensor Shape** | `(1, 32, 32)` | `(1, 28, 28)` |
| **Implementation** | Manual Tensor Conversion | `torchvision.transforms` |

---

## Detailed Analysis

### 1. `CNN_pickle.py` Pipeline

This script relies on "Offline Preprocessing," meaning distinct work was done *before* the training script even runs.

**A. Offline Preparation:**
*   **Resizing:** Images were resized to **32x32** when creating the pickle file.
*   **Denoising:** A cleaning script (`pickle_clean.py`) processed the data, setting any pixel value < 50 to 0. This removes background noise.

**B. Online Loading (During Training):**
Inside `DigitsDataset.__getitem__`:
1.  **Loading:** Loads a 32x32x1 array from the pickle dictionary.
2.  **Scaling:** Manually converts to `float32` and divides by 255.0.
    ```python
    image = torch.tensor(image, dtype=torch.float32) / 255.0
    ```
3.  **Reshaping:** Permutes dimensions to match PyTorch's Channel-First expectation.
    ```python
    image = image.permute(2, 0, 1) # (32, 32, 1) -> (1, 32, 32)
    ```

**Result:** The model receives a **clean, denoised 32x32** image.

### 2. `CNN_raw.py` Pipeline

This script performs "Online Preprocessing" exclusively, starting from the raw image files on disk.

**A. Online Loading:**
Inside `CharacterDataset.__getitem__` & `transforms`:
1.  **Loading:** Reads raw image files using OpenCV (Grayscale).
2.  **Resizing:** `transforms.Resize((28, 28))` forces the image to **28x28**.
3.  **Scaling & Shaping:** `transforms.ToTensor()` converts the PIL image to a tensor and scales values to [0, 1].

**Result:** The model receives a **potentially noisy 28x28** image (depending on the raw source quality), as no thresholding/denoising step is applied.

## Key Takeaways for Model Learning

1.  **Resolution Mismatch:** A model trained with `CNN_pickle.py` expects 32x32 inputs, while `CNN_raw.py` expects 28x28. They are **not compatible** without resizing inputs.
2.  **Noise Sensitivity:** `CNN_pickle.py` trains on cleaner data. If deployed on raw, noisy images without the same `< 50` thresholding step, it might struggle with background artifacts. `CNN_raw.py` might be more robust to noise (as it sees it during training) or less accurate (if the noise overwhelms the signal), depending on the dataset quality.
