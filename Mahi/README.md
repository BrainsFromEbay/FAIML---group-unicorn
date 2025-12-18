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

All trained models are saved in `Mahi/preprocessed/models/`.

---

### Part 1: Preprocessing Utils
Located in `Mahi/preprocessed/data_preprocessing/`.

1.  **`raw_image_to_binary.py`**:
    - **Input**: `Train/` and `Validation/` directories (Raw Images).
    - **Output**: `Mahi/preprocessed/digits_data.pickle` (32x32 binary data).
2.  **`pickle_clean.py`**:
    - **Input**: `digits_data.pickle`.
    - **Process**: Removes noise (threshold < 50).
    - **Output**: `Mahi/preprocessed/digits_data_cleaned.pickle`.

---

### Part 2: Training Workflows
Located in `Mahi/preprocessed/train_script/`.

#### A. From Raw Images (28x28)
- **`CNN_raw.py`**:
    - Trains a simple CNN directly on the raw image directories (resized to 28x28).
    - **Output**: `Mahi/preprocessed/models/CNN_raw.pth`.

#### B. From Pickled Data (32x32)
- **`CNN_pickle.py`**: Trains a CNN on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/preprocessed/models/CNN_pickle.pth`.
- **`MLP_pickle.py`**: Trains an MLP on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/preprocessed/models/mlp_pickle.pth`.
- **`random_forrest.py`**: Trains a Random Forest on `digits_data_cleaned.pickle`.
    - **Output**: `Mahi/preprocessed/models/rf.joblib`.

#### C. From MNIST Data (28x28) - **Recommended**
These models generalize best to external handwritten digits (`custom_test`).
- **`CNN_mnist.py`**: Trains a CNN on the official MNIST dataset.
    - **Output**: `Mahi/preprocessed/models/cnn_mnist.pth`.
- **`MLP_mnist.py`**: Trains an MLP on the official MNIST dataset.
    - **Output**: `Mahi/preprocessed/models/mlp_mnist.pth`.
- **`random_forrest_mnist.py`**: Trains a Random Forest on the official MNIST dataset.
    - **Output**: `Mahi/preprocessed/models/rf_mnist.joblib`.

---

### Part 3: Inference Workflows
Located in `Mahi/preprocessed/inference/`.

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
- `Mahi/preprocessed/data_preprocessing/`: Scripts to create/clean pickle data.
- `Mahi/preprocessed/train_script/`: All active training scripts.
- `Mahi/preprocessed/inference/`: All inference/testing scripts.
- `Mahi/preprocessed/models/`: Directory where all trained models are saved.
- `Mahi/preprocessed/`: Contains the large `.pickle` datasets.

