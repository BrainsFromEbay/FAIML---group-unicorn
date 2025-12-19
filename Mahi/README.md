# Mahi Talukder:

## Dataset

The dataset used for training can be found [on Kaggle](https://www.kaggle.com/datasets/vaibhao/handwritten-characters).

The expected directory structure for the dataset is as follows:

```
├───Train/
│   ├───@/
│   ├───├───filename.png
│   ├───├───...png
│   ├───&/
│   ├───#/
│   ├───$/
│   ├───0/
│   ├───1/
│   ├───2/
│   ├───3/
│   ├───4/
│   ├───5/
│   ├───6/
│   ├───7/
│   ├───8/
│   ├───9/
│   ├───A/
│   ├───...
│   ├───...
│   └───Z/
└───Validation/
    ├───@/
    ├───├───filenanme.png¨
    ├───├───...png
    ├───&/
    ├───#/
    ├───$/
    ├───0/
    ├───1/
    ├───2/
    ├───3/
    ├───4/
    ├───5/
    ├───6/
    ├───7/
    ├───8/
    ├───9/
    ├───A/
    ├───...
    ├───...
    └───Z/
```

## Setup and Installation

This project uses Python and several dependencies. You can set up the environment using either `pip` with `requirements.txt` or `conda` with `environment.yml`.

### Using Conda (Recommended)

To create a Conda environment with all the necessary dependencies, run the following command from the project's root directory:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment named `thesis`. To activate it, use:

```bash
conda activate thesis
```

### Using Pip

If you are not using Conda, you can install the required packages using pip. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Workflows

This project contains several distinct workflows for handling and training models on image data. They are organized into three main categories based on the data source: **Raw Images**, **Pickled Data**, and **MNIST Data**.

All trained models are saved in `Mahi/src/models/`.

---

### Part 1: Preprocessing Utils
Located in `Mahi/src/data_preprocessing/`.

1.  **`raw_image_to_binary.py`**:
    - **Input**: `Train/` and `Validation/` directories (Raw Images).
    - **Output**: `Mahi/src/digits_data.pickle` (32x32 binary data).
2.  **`pickle_clean.py`**:
    - **Input**: `digits_data.pickle`.
    - **Process**: Removes noise (threshold < 50).
    - **Output**: `Mahi/src/digits_data_cleaned.pickle`.

---

### Part 2: Training Workflows
Located in `Mahi/src/train_script/`.

#### A. From Raw Images (28x28)
- **`CNN_raw.py`**:
    - Trains a simple CNN directly on the raw image directories (resized to 28x28).
    - **Output**: `Mahi/src/models/CNN_raw.pth`.

#### B. From Pickled Data (32x32)
- **`CNN_pickle.py`**: Trains a CNN on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/src/models/CNN_pickle.pth`.
- **`MLP_pickle.py`**: Trains an MLP on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/src/models/mlp_pickle.pth`.
- **`random_forrest.py`**: Trains a Random Forest on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/src/models/rf.joblib`.

#### C. From MNIST Data (28x28) - **Recommended**
These models generalize best to external handwritten digits (`custom_test`).
- **`CNN_mnist.py`**: Trains a CNN on the official MNIST dataset.
    - **Output**: `Mahi/src/models/cnn_mnist.pth`.
- **`MLP_mnist.py`**: Trains an MLP on the official MNIST dataset.
    - **Output**: `Mahi/src/models/mlp_mnist.pth`.
- **`random_forrest_mnist.py`**: Trains a Random Forest on the official MNIST dataset.
    - **Output**: `Mahi/src/models/rf_mnist.joblib`.

---

### Part 3: Inference Workflows
Located in `Mahi/src/inference/`.

Each training script has a corresponding inference script to test the model on the `custom_test` directory.

- **For Raw Models**: `cnn_raw_inference.py`.
- **For Pickle Models**: `cnn_pickle_inference.py`, `mlp_pickle_inference.py`, `rf_inference.py`.
- **For MNIST Models**: `cnn_mnist_inference.py`, `mlp_mnist_inference.py`, `rf_mnist_inference.py`.

---

## File Descriptions

- `README.md`: This file.
- `Mahi/environment.yml`: Conda environment dependencies.
- `Mahi/requirements.txt`: Pip package requirements.
- `Mahi/raw_image/MLP_from_raw_image.py`: Legacy script for MLP on raw images.

### Key Directories
- `Mahi/src/data_preprocessing/`: Scripts to create/clean pickle data.
- `Mahi/src/train_script/`: All active training scripts.
- `Mahi/src/inference/`: All inference/testing scripts.
- `Mahi/src/models/`: Directory where all trained models are saved.
- `Mahi/src/`: Contains the large `.pickle` datasets.

---

# Model Performance Comparison Report

This report summarizes the performance of the active models in `Mahi/src/inference/`.
All models were re-evaluated on the `custom_test` dataset (20 images: 10 'normal' x.png and 10 'noisy' x(1).png).

## Prediction Matrix (Full Test Set)

| Image | Exp | CNN Raw | CNN Pickle | CNN MNIST | MLP Pickle | MLP MNIST | RF Pickle | RF MNIST |
| :---: | :-: | :-----: | :--------: | :-------: | :--------: | :-------: | :-------: | :------: |
| 0(1).png | 0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| 0.png    | 0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 2 ❌ | 0 ✅ |
| 1(1).png | 1 | 9 ❌ | 1 ✅ | 1 ✅ | 4 ❌ | 1 ✅ | 7 ❌ | 3 ❌ |
| 1.png    | 1 | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | 2 ❌ | 1 ✅ |
| 2(1).png | 2 | 2 ✅ | 2 ✅ | 2 ✅ | 1 ❌ | 2 ✅ | 1 ❌ | 8 ❌ |
| 2.png    | 2 | 2 ✅ | 7 ❌ | 2 ✅ | 2 ✅ | 2 ✅ | 2 ✅ | 2 ✅ |
| 3(1).png | 3 | 3 ✅ | 3 ✅ | 3 ✅ | 3 ✅ | 3 ✅ | 3 ✅ | 7 ❌ |
| 3.png    | 3 | 3 ✅ | 3 ✅ | 3 ✅ | 9 ❌ | 3 ✅ | 7 ❌ | 1 ❌ |
| 4(1).png | 4 | 4 ✅ | 4 ✅ | 7 ❌ | 4 ✅ | 7 ❌ | 7 ❌ | 7 ❌ |
| 4.png    | 4 | 1 ❌ | 7 ❌ | 1 ❌ | 4 ✅ | 1 ❌ | 2 ❌ | 5 ❌ |
| 5(1).png | 5 | 5 ✅ | 5 ✅ | 5 ✅ | 5 ✅ | 5 ✅ | 5 ✅ | 5 ✅ |
| 5.png    | 5 | 5 ✅ | 3 ❌ | 5 ✅ | 5 ✅ | 1 ❌ | 5 ✅ | 5 ✅ |
| 6(1).png | 6 | 6 ✅ | 6 ✅ | 6 ✅ | 6 ✅ | 6 ✅ | 6 ✅ | 2 ❌ |
| 6.png    | 6 | 4 ❌ | 0 ❌ | 8 ❌ | 9 ❌ | 6 ✅ | 1 ❌ | 6 ✅ |
| 7(1).png | 7 | 7 ✅ | 7 ✅ | 7 ✅ | 4 ❌ | 1 ❌ | 7 ✅ | 1 ❌ |
| 7.png    | 7 | 7 ✅ | 7 ✅ | 7 ✅ | 7 ✅ | 7 ✅ | 7 ✅ | 2 ❌ |
| 8(1).png | 8 | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ |
| 8.png    | 8 | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ | 8 ✅ | 1 ❌ | 2 ❌ |
| 9(1).png | 9 | 9 ✅ | 9 ✅ | 3 ❌ | 3 ❌ | 7 ❌ | 7 ❌ | 7 ❌ |
| 9.png    | 9 | 9 ✅ | 9 ✅ | 9 ✅ | 9 ✅ | 9 ✅ | 1 ❌ | 8 ❌ |

## Accuracy Scoreboard (All 20 Images)

| Rank | Model Script | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
| 1 | `cnn_raw_inference.py` | 17 | 20 | **85.0%** |
| 2 | `cnn_mnist_inference.py` | 16 | 20 | **80.0%** |
| 2 | `cnn_pickle_inference.py` | 16 | 20 | **80.0%** |
| 4 | `mlp_mnist_inference.py` | 15 | 20 | **75.0%** |
| 5 | `mlp_pickle_inference.py` | 14 | 20 | **70.0%** |
| 6 | `rf_inference.py` | 9 | 20 | **45.0%** |
| 7 | `rf_mnist_inference.py` | 8 | 20 | **40.0%** |

## Analysis

1.  **Top Performers**: `CNN Raw` remains the champion (**85%**), but `CNN Pickle` and `CNN MNIST` are surprisingly close runners-up (**80%**).
    - **Note on CNN Pickle**: It performed **perfectly** on the 10 "noisy" `(1).png` images, even though it struggled with some clean ones. This suggests the "noise" (preprocessing artifact) might actually match the distribution of the pickled training data better.
2.  **Noisy Images**: Most models handled the `(1).png` images very well, often better than the clean ones. This might be due to the thresholding/inversion preprocessing steps aligning better with how the models were trained (especially for `Pickle` models).
3.  **Specific Digits**:
    - **'0'**: `CNN Pickle`, `CNN Raw`, `CNN MNIST`, and `MLP Pickle` all nailed '0' perfectly across both variants.
    - **'4'**: Extremely difficult. `MLP Pickle` was the **only** model to correctly identify the clean `4.png`. `CNN Raw` and others consistently failed on it.
    - **'1(1)'**: `CNN Raw` oddly failed on this (predicted 9), while `CNN Pickle` and `CNN MNIST` got it right.

## Conclusion

The CNN architecture is superior for this task, regardless of the training data source (Raw, Pickle, or MNIST), consistently scoring 80-85%. Random Forests are significantly weaker (40-45%).
For the best results, an **ensemble of CNN Raw + CNN Pickle + MLP Pickle** (which is the only one that knows '4') would likely achieve near perfect accuracy.

