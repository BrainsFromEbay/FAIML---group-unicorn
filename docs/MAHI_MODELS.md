# Mahi's Collection of Models

This document provides an overview of the various models implemented by Mahi. This contribution is a comprehensive exploration of different modeling techniques for handwritten digit recognition.

The code for these models is located in the `Mahi/` directory, which is structured into `src` for source code and `results` for model performance metrics.

## Overview of Models

Mahi has implemented and trained several types of models, including:

*   **Convolutional Neural Networks (CNNs)**
*   **Multi-Layer Perceptrons (MLPs)**
*   **Random Forest Classifiers**

Each of these models has been trained on different variations of the data, leading to multiple saved model files. The naming convention of the files indicates the model type and the data it was trained on (e.g., `CNN_pickle.pth`, `MLP_mnist.py`, `rf_mnist.joblib`).

## Directory Structure

The `Mahi/src/` directory is organized as follows:

*   `data_preprocessing/`: Contains scripts for cleaning and preparing the data.
*   `train_script/`: Includes the Python scripts used to train each of the models.
*   `inference/`: Contains scripts that are used to load the trained models and make predictions. These scripts are called by the main `gui.py`.
*   `models/`: This directory stores the saved model files (`.pth` for PyTorch models, `.joblib` for scikit-learn models).

## Model Details

### Convolutional Neural Networks (CNNs)

*   **Description:** Deep learning models with architectures similar to Jere's CNN, designed to learn spatial hierarchies of features from the input images.
*   **Variations:**
    *   `CNN_raw`: Trained on raw image data.
    *   `CNN_pickle`: Trained on a pickled version of the data.
*   **Training Scripts:** `CNN_raw.py`, `CNN_pickle.py`
*   **Inference Scripts:** `cnn_raw_inference.py`, `cnn_pickle_inference.py`

### Multi-Layer Perceptrons (MLPs)

*   **Description:** A classic type of feedforward artificial neural network.
*   **Variations:**
    *   `MLP_mnist`: Trained on the standard MNIST dataset.
    *   `MLP_pickle`: Trained on a pickled version of the data.
*   **Training Scripts:** `MLP_mnist.py`, `MLP_pickle.py`
*   **Inference Scripts:** `mlp_mnist_inference.py`, `mlp_pickle_inference.py`

### Random Forest

*   **Description:** An ensemble learning method that operates by constructing a multitude of decision trees at training time.
*   **Variations:**
    *   `random_forrest_mnist`: Trained on the standard MNIST dataset.
*   **Training Script:** `random_forrest_mnist.py`
*   **Inference Script:** `rf_mnist_inference.py`

## Prediction and Integration

Each model has a corresponding inference script in the `Mahi/src/inference/` directory. These scripts follow a similar pattern:

*   A `load_model()` function that loads the saved model file.
*   A `preprocess_image()` function to prepare the input image for the model.
*   A `predict_single()` or similar function to perform the prediction.

These scripts are imported by the main `gui.py` to allow the user to select and test any of Mahi's models.
