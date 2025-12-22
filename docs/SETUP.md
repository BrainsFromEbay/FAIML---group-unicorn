# Setup and Installation

This guide will walk you through setting up the necessary environment to run the Handwritten Digit Recognition project. There are two primary ways to set up the environment: using `pip` with `requirements.txt` or using Conda with `environment.yml`.

## Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)
*   `conda` (if you choose the Conda-based setup)

## Option 1: Using `pip` (Recommended)

This is the recommended method for setting up the project's dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Option 2: Using Conda

If you prefer to use Conda, you can use the `environment.yml` file to create a Conda environment.

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate thesis
    ```

## Running the Application

Once you have installed the dependencies using either of the methods above, you can run the main application.

The main application is the Graphical User Interface (GUI), which you can start with the following command:

```bash
streamlit run gui.py
```

This will start a local web server and open the application in your browser.

## Downloading the Dataset

The models in this project are trained on the MNIST dataset. While some of the training scripts may download the dataset automatically, it's recommended to have it available locally. You should place the MNIST dataset files in the `data/MNIST/` directory.
