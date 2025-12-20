# Feature Extraction and Data Preprocessing

"Feature Extraction" sounds complicated, but it just means **preparing the data** so the computer can understand it easily. We don't just dump raw images into the model; we clean and shape them first.

---

## 1. Shaping the Input (The "Container")

Before we look at the pixels, we need to decide what shape the data should be in.

### A. For CNNs (The 3D block)
*   **Shape**: `(Channels, Height, Width)`
*   **Our Data**: `(1, 28, 28)` or `(1, 32, 32)`
*   **Explanation**:
    *   **1 Channel**: Because our images are **Grayscale** (Black & White). Color images would have 3 channels (Red, Green, Blue).
    *   **28x28**: The standard size for MNIST digits.
    *   **32x32**: Used in our custom pickled dataset for slightly higher detail.
*   **Why?** CNNs need this 3D shape to slide their filters over the height and width.

### B. For MLPs and Random Forest (The Flat List)
*   **Shape**: `(Height * Width)`
*   **Our Data**: A list of `784` numbers (for 28x28) or `1024` numbers (for 32x32).
*   **Process**: We take the square image and "unroll" it row by row into a single long line.
*   **Why?** These algorithms don't have "eyes" to see a grid; they only understand lists of numbers.

---

## 2. Normalization (The "Scaler")

Raw pixel values are usually numbers between **0 (Black)** and **255 (White)**.
Computers, however, learn much faster with small numbers like 0 to 1.

### What we do:
1.  **Divide by 255.0**:
    *   This converts the range `[0, 255]` into `[0, 1]`.
    *   Used in `CNN_pickle.py`, `random_forrest.py`.
2.  **Standardization (Mean & Std Dev)**:
    *   Sometimes we go further and force the numbers to be centered around 0, roughly between `[-1, 1]`.
    *   Example from `CNN_mnist.py`: `transforms.Normalize((0.5,), (0.5,))`
    *   **Why?** This keeps the math inside the neural network stable, preventing numbers from getting too huge or too small to calculate.

---

## 3. Data Augmentation (The "Teacher")

If we only teach the student how to read a perfect "7", they might fail if they see a "7" that is slightly tilted or shifted.
**Augmentation** artificially creates "messy" versions of our images to make the model smarter.

### Techniques Used:
*   **Random Affine / Rotation**:
    *   We rotate the image slightly (e.g., ±15 degrees).
    *   We shift it up/down or left/right (Translation).
    *   We zoom in or out slightly (Scale).
*   **Code Example**:
    ```python
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ```
*   **Result**: The model learns that a "7" is still a "7" even if it's crooked or off-center.

---

## 4. Data Sources

### A. Raw Folders (`CNN_raw.py`)
*   **Method**: `cv2.imread` + `Image.fromarray`
*   **Process**: Reads image files (JPG/PNG) directly from folders on the hard drive. Good for simple tests but slow for big datasets.

### B. Pickle Files (`CNN_pickle.py`)
*   **Method**: `pickle.load`
*   **Process**: We pre-converted all images into a big efficient Numpy array and saved it as a `.pickle` file.
*   **Benefit**: This is **much faster** to load because the computer doesn't have to open thousands of individual tiny files; it just loads one big block of data memory.
