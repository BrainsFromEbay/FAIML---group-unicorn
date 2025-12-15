# Jere Villman:

---

# Tasnuba Oyshe:

---

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

This project contains two distinct and currently incompatible workflows for handling image data.

---

### Workflow 1: Direct PNG Training (28x28 Images)

This workflow trains a CNN model directly from PNG images of digits.

- **Script:** `Mahi/digits_CNN_from_png.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **28x28 pixels**, and uses them to train a `SimpleCNN` model. The dataset it was trained on has:
  - Training samples: 416126
  - Validation samples: 11453
- **Output:** The trained model is saved as `Mahi/digits_model_CNN__from_png.pth`.

#### Using the 28x28 Model

The saved `digits_model_CNN__from_png.pth` can be used for inference on new images. The key steps are to define the `SimpleCNN` architecture from the script and then load the state dictionary into it.

---

### Workflow 2: Preprocessing to Pickle (32x32 Images)

This workflow preprocesses the PNG images and saves them into a single binary file.

- **Script:** `Mahi/raw_image_to_binary.py`
- **Process:** The script reads images from `Train/` and `Validation/` directories, resizes them to **32x32 pixels**, and saves the data into a pickle file.
- **Output:** A file named `digits_data.pickle` containing the processed image data and labels.
- **Inspection:** The `Mahi/inspect.ipynb` Jupyter Notebook can be used to load and visualize the data from `digits_data.pickle`.

the digits_data.pickle file is too big to be uploaded by git push. Please download the file from [onedrive](google.com)

---

## File Descriptions

- `README.md`: This file.
- `database_structure.md`: A file describing the expected dataset structure.
- `Mahi/environment.yml`: Conda environment dependencies.
- `Mahi/requirements.txt`: Pip package requirements.
- `Mahi/digits_CNN_from_png.py`: Main script for **Workflow 1**. Trains the model on 28x28 PNGs.
- `Mahi/digits_model_CNN__from_png.pth`: The trained model file from **Workflow 1**.
- `Mahi/raw_image_to_binary.py`: Preprocessing script for **Workflow 2**. Generates a pickle file with 32x32 images.
- `Mahi/digits_data.pickle`: The output of `raw_image_to_binary.py`.
- `Mahi/inspect.ipynb`: A Jupyter Notebook to inspect the `digits_data.pickle` file.
