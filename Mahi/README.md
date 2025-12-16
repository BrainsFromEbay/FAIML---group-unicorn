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

These workflows read PNG files directly, resize them to 28x28, and train simple models.

#### Workflow 1: CNN model from raw image
- **Script:** `Mahi/digits_CNN_from_raw_image.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **28x28 pixels**, and uses them to train a `SimpleCNN` model. The dataset it was trained on has:
  - Training samples: 416126
  - Validation samples: 11453
- **Output:** The trained model is saved as `Mahi/digits_model_CNN_from_raw_image.pth`.

#### Workflow 2: MLP model from raw image
- **Script:** `Mahi/digits_model_MLP_from_raw_image.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **28x28 pixels**, and uses them to train a `SimpleMLP` model. This workflow focuses only on digits 0-9. The dataset it was trained on has:
  - Training samples: 86968
  - Validation samples: 11453
- **Output:** The trained model is saved as `Mahi/digits_model_MLP_from_raw_image.pth`.

---

### Part 2: Workflows using Pickled Data (32x32)

This is a more advanced pipeline that involves preprocessing the images into a binary format, cleaning the data, and then training a more robust model.

#### Workflow 3: Preprocessing to Pickle
- **Script:** `Mahi/raw_image_to_binary.py`
- **Process:** Reads digit images (0-9) from `Train/` and `Validation/` directories, resizes them to **32x32 pixels**, and saves them into a single binary file.
- **Output:** A file named `digits_data.pickle` containing the processed image data and labels for digits.
- **Note:** The `digits_data.pickle` file is too large for Git. Please download it from [Google Drive](https://drive.google.com/file/d/1JJ9WHbLCZDNVymFtcXM1cAB6A6tGdMk6/view?usp=sharing) if needed.

#### Workflow 4: Cleaning the Pickle File
- **Script:** `Mahi/pickle_clean.py`
- **Process:** This script loads `digits_data.pickle`, removes noise by setting all pixel intensity values below a threshold (50) to zero, and saves the result in a new file. This helps the model focus on the important features of the digits.
- **Input:** `Mahi/digits_data.pickle`
- **Output:** `Mahi/digits_data_cleaned.pickle`

#### Workflow 5: CNN from Cleaned Pickle Data
- **Script:** `Mahi/CNN_from_pickle.py`
- **Process:** This is the most advanced training script. It loads the cleaned data, uses an efficient PyTorch `Dataset` for on-the-fly normalization, and trains a `SimpleDigitCNN` with mixed-precision to reduce VRAM usage.
- **Input:** `Mahi/digits_data_cleaned.pickle`
- **Output:** 
  - `best_digit_model.pth`: The model state dictionary that achieved the highest validation accuracy during training.
  - `simple_digit_cnn_full.pth`: The complete, saved PyTorch model object from the last epoch.

---

## File Descriptions

- `README.md`: This file.
- `Mahi/environment.yml`: Conda environment dependencies.
- `Mahi/requirements.txt`: Pip package requirements.

- **Workflow Scripts:**
  - `Mahi/digits_CNN_from_raw_image.py`: (Workflow 1) Trains a CNN on 28x28 raw PNGs of all characters.
  - `Mahi/digits_model_MLP_from_raw_image.py`: (Workflow 2) Trains an MLP on 28x28 raw PNGs of digits.
  - `Mahi/raw_image_to_binary.py`: (Workflow 3) Preprocesses 32x32 digit images into `digits_data.pickle`.
  - `Mahi/pickle_clean.py`: (Workflow 4) Cleans `digits_data.pickle` and saves `digits_data_cleaned.pickle`.
  - `Mahi/CNN_from_pickle.py`: (Workflow 5) Trains a CNN on the cleaned 32x32 pickle data.

- **Model Files:**
  - `Mahi/digits_model_CNN_from_raw_image.pth`: Trained model from **Workflow 1**.
  - `Mahi/digits_model_MLP_from_raw_image.pth`: Trained model from **Workflow 2**.
  - `Mahi/best_digit_model.pth`: Best-performing model state from **Workflow 5**.
  - `Mahi/simple_digit_cnn_full.pth`: Full model saved at the end of **Workflow 5**.

- **Data Files:**
  - `Mahi/digits_data.pickle`: The output of `raw_image_to_binary.py`. Contains 32x32 digit images.
  - `Mahi/digits_data_cleaned.pickle`: The output of `pickle_clean.py`. A denoised version of the above.

- **Utilities:**
  - `Mahi/inspect.ipynb`: A Jupyter Notebook to inspect and visualize the data from the `.pickle` files.