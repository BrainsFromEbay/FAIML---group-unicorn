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

This project contains several distinct workflows for handling and training models on image data. They are grouped into two main categories: training directly from raw image files and a more advanced pipeline that uses preprocessed pickle files.

---

### Part 1: Workflows from Raw Images (28x28)

These workflows read PNG files directly, resize them to 28x28, and train simple models. The scripts and models are located in the `Mahi/raw_image/` directory.

#### Workflow 1: CNN model from raw image

- **Script:** `Mahi/raw_image/CNN_from_raw_image.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **28x28 pixels**, and uses them to train a `SimpleCNN` model. This workflow focuses only on digits 0-9. The dataset it was trained on has:
  - Training samples: 86968
  - Validation samples: 11453
- **Output:** The trained model is saved as `Mahi/raw_image/CNN_from_raw_image.pth`.

#### Workflow 2: MLP model from raw image

- **Script:** `Mahi/raw_image/MLP_from_raw_image.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **28x28 pixels**, and uses them to train a `SimpleMLP` model. This workflow focuses only on digits 0-9. The dataset it was trained on has:
  - Training samples: 86968
  - Validation samples: 11453
- **Output:** The trained model is saved as `Mahi/raw_image/MLP_from_raw_image.pth`.

---

### Part 2: Workflows using Pickled Data (32x32)

This is a more advanced pipeline that involves preprocessing the images into a binary format, cleaning the data, and then training a more robust model. These files are located in `Mahi/preprocessed/`.

#### Workflow 3: Preprocessing to Pickle

- **Script:** `Mahi/preprocessed/raw_image_to_binary.py`
- **Process:** Reads digit images (0-9) from `Train/` and `Validation/` directories, resizes them to **32x32 pixels**, and saves them into a single binary file.
- **Output:** A file named `Mahi/preprocessed/digits_data.pickle` containing the processed image data and labels for digits.
- **Note:** The `digits_data.pickle` file is too large for Git. Please download it from [Google Drive](https://drive.google.com/file/d/1JJ9WHbLCZDNVymFtcXM1cAB6A6tGdMk6/view?usp=sharing) if needed.

#### Workflow 4: Cleaning the Pickle File

- **Script:** `Mahi/preprocessed/pickle_clean.py`
- **Process:** This script loads `Mahi/preprocessed/digits_data.pickle`, removes noise by setting all pixel intensity values below a threshold (50) to zero, and saves the result in a new file. This helps the model focus on the important features of the digits.
- **Input:** `Mahi/preprocessed/digits_data.pickle`
- **Output:** `Mahi/preprocessed/digits_data_cleaned.pickle`

#### Workflow 5: CNN from Cleaned Pickle Data

- **Script:** `Mahi/preprocessed/CNN_from_pickle.py`
- **Process:** This is the most advanced training script. It loads the cleaned data, uses an efficient PyTorch `Dataset` for on-the-fly normalization, and trains a `SimpleDigitCNN` with mixed-precision to reduce VRAM usage.
- **Input:** `Mahi/preprocessed/digits_data_cleaned.pickle`
- **Output:**
  - `best_digit_model.pth`: The model state dictionary that achieved the highest validation accuracy during training (saved in the project root).
  - `Mahi/preprocessed/CNN_digit_full.pth`: The complete, saved PyTorch model object from the last epoch.

#### Workflow 6: MLP from Cleaned Pickle Data

- **Script:** `Mahi/preprocessed/MLP_from_pickle.py`
- **Process:** This script loads the cleaned data from `Mahi/preprocessed/digits_data_cleaned.pickle` and trains a `SimpleMLP` model. It flattens the 32x32 images and uses a series of fully connected layers. Like the CNN workflow, it uses mixed-precision for efficiency.
- **Input:** `Mahi/preprocessed/digits_data_cleaned.pickle`
- **Output:**
  - `Mahi/preprocessed/best_mlp_model.pth`: The model state dictionary that achieved the highest validation accuracy during training.
  - `Mahi/preprocessed/mlp_full.pth`: The complete, saved PyTorch model object from the last epoch.

#### Workflow 7: Random Forest from Cleaned Pickle Data

- **Script:** `Mahi/preprocessed/random_forrest.py`
- **Process:** This script loads the cleaned data from `Mahi/preprocessed/digits_data_cleaned.pickle`, flattens the 32x32 images, and trains a `RandomForestClassifier` model using Scikit-learn. It uses all available CPU cores for efficiency.
- **Input:** `Mahi/preprocessed/digits_data_cleaned.pickle`
- **Output:**
  - `Mahi/preprocessed/rf_model.joblib`: The complete, saved Scikit-learn model object.

---

## File Descriptions

- `README.md`: This file.
- `Mahi/environment.yml`: Conda environment dependencies.
- `Mahi/requirements.txt`: Pip package requirements.

- **Workflow Scripts:**

  - `Mahi/raw_image/CNN_from_raw_image.py`: (Workflow 1) Trains a CNN on 28x28 raw PNGs of digits.
  - `Mahi/raw_image/MLP_from_raw_image.py`: (Workflow 2) Trains an MLP on 28x28 raw PNGs of digits.
  - `Mahi/preprocessed/raw_image_to_binary.py`: (Workflow 3) Preprocesses 32x32 digit images into `Mahi/preprocessed/digits_data.pickle`.
  - `Mahi/preprocessed/pickle_clean.py`: (Workflow 4) Cleans `Mahi/preprocessed/digits_data.pickle` and saves `Mahi/preprocessed/digits_data_cleaned.pickle`.
  - `Mahi/preprocessed/CNN_from_pickle.py`: (Workflow 5) Trains a CNN on the cleaned 32x32 pickle data.
  - `Mahi/preprocessed/MLP_from_pickle.py`: (Workflow 6) Trains an MLP on the cleaned 32x32 pickle data.
  - `Mahi/preprocessed/random_forrest.py`: (Workflow 7) Trains a Random Forest classifier on the cleaned 32x32 pickle data.

- **Model Files:**

  - `Mahi/raw_image/CNN_from_raw_image.pth`: Trained model from **Workflow 1**.
  - `Mahi/raw_image/MLP_from_raw_image.pth`: Trained model from **Workflow 2**.
  - `best_digit_model.pth`: Best-performing model state from **Workflow 5** (saved in project root).
  - `Mahi/preprocessed/CNN_digit_full.pth`: Full model saved at the end of **Workflow 5**.
  - `Mahi/preprocessed/best_mlp_model.pth`: Best-performing model state from **Workflow 6**.
  - `Mahi/preprocessed/mlp_full.pth`: Full model saved at the end of **Workflow 6**.
  - `Mahi/preprocessed/rf_model.joblib`: Trained Random Forest model from **Workflow 7**.

- **Data Files:**

  - `Mahi/preprocessed/digits_data.pickle`: The output of `raw_image_to_binary.py`. Contains 32x32 digit images.
  - `Mahi/preprocessed/digits_data_cleaned.pickle`: The output of `pickle_clean.py`. A denoised version of the above.

- **Utilities:**
  - `Mahi/inspect.ipynb`: A Jupyter Notebook to inspect and visualize the data from the `.pickle` files.
