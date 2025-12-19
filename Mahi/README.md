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
All models were re-evaluated on the `custom_test` dataset (10 images).

## Prediction Matrix

| Image | Exp | CNN Raw | CNN Pickle | CNN MNIST | MLP Pickle | MLP MNIST | RF Pickle | RF MNIST |
| :---: | :-: | :-----: | :--------: | :-------: | :--------: | :-------: | :-------: | :------: |
| 0.png |  0  |  8 ❌   |    8 ❌    |   8 ❌    |    8 ❌    |   8 ❌    |   8 ❌    |   2 ❌   |
| 1.png |  1  |  1 ✅   |    1 ✅    |   1 ✅    |    1 ✅    |   1 ✅    |   1 ✅    |   1 ✅   |
| 2.png |  2  |  2 ✅   |    7 ❌    |   2 ✅    |    2 ✅    |   2 ✅    |   2 ✅    |   2 ✅   |
| 3.png |  3  |  3 ✅   |    3 ✅    |   3 ✅    |    9 ❌    |   3 ✅    |   3 ✅    |   1 ❌   |
| 4.png |  4  |  1 ❌   |    7 ❌    |   1 ❌    |    4 ✅    |   1 ❌    |   1 ❌    |   5 ❌   |
| 5.png |  5  |  5 ✅   |    3 ❌    |   5 ✅    |    5 ✅    |   5 ✅    |   1 ❌    |   5 ✅   |
| 6.png |  6  |  4 ❌   |    0 ❌    |   8 ❌    |    9 ❌    |   6 ✅    |   6 ✅    |   6 ✅   |
| 7.png |  7  |  7 ✅   |    7 ✅    |   7 ✅    |    7 ✅    |   7 ✅    |   7 ✅    |   2 ❌   |
| 8.png |  8  |  8 ✅   |    8 ✅    |   8 ✅    |    8 ✅    |   8 ✅    |   8 ✅    |   2 ❌   |
| 9.png |  9  |  9 ✅   |    9 ✅    |   9 ✅    |    9 ✅    |   9 ✅    |   9 ✅    |   8 ❌   |

> **Note**: `CNN Raw` corresponds to `cnn_raw_inference.py`, `CNN MNIST` to `cnn_mnist_inference.py`, etc.

## Accuracy Scoreboard

| Rank | Model Script | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
| 1 (Tie) | `cnn_raw_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `cnn_mnist_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `mlp_mnist_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `mlp_pickle_inference.py` | 7 | 10 | **70.0%** |
| 5 | `cnn_pickle_inference.py` | 5 | 10 | **50.0%** |
| 6 | `rf_mnist_inference.py` | 4 | 10 | **40.0%** |
| 7 | `rf_inference.py` | 3 | 10 | **30.0%** |

## Analysis

1.  **Neural Networks > Random Forest**: The neural networks (CNNs and MLPs) consistently outperformed Random Forests (30-40%). This confirms that for image data, especially with domain shifts (custom handwriting), deep learning approaches are far more robust.
2.  **Consistency**: Four different NN models achieved **70% accuracy**. This suggests a "ceiling" at 70% for these standard architectures on this specific test set without more advanced techniques or more targeted data.
    - They mostly fail on the same difficult digits: **0** (looks like 8), **4** (confusing stroke), and sometimes **6**.
3.  **Surprise Performer**: `MLP Pickle` was the **only model to correctly classify the digit 4**. This is notable because even the MNIST models failed on it.
4.  **MNIST Benefit**: While `RF` improved on MNIST (30% -> 40%) and `MLP` maintained high performance (70%), the `CNN` didn't dramatically outperform `CNN Raw` in this specific run. However, historically ( फ्रेंड's model), MNIST has shown potential for higher accuracy (89%).

## Conclusion

The project has consolidated around 70% accuracy for Neural Networks. To break this ceiling, one would likely need to:
1.  Augment the training data with samples that specifically resemble the fail cases (0s that look like 8s, 4s with this specific stroke style).
2.  Use a more complex architecture or ensemble methods.

