# Models

This project implements and compares several different machine learning models for handwritten digit recognition. The models are contributed by different team members and are organized in their respective directories.

This page provides a brief overview of the models. For more detailed information about each model, please refer to the specific documentation pages linked below.

## Model Overview

The following models are included in this project:

*   **Convolutional Neural Network (CNN)** by Jere
    *   A deep learning model built with PyTorch.
    *   [More details](./JERE_CNN.md)

*   **HOG + Logistic Regression** by Oyshe
    *   A classical machine learning model that uses Histogram of Oriented Gradients (HOG) as features.
    *   [More details](./OYSHE_HOG.md)

*   **A Collection of Models** by Mahi
    *   This includes several models:
        *   Convolutional Neural Networks (CNNs)
        *   Multi-Layer Perceptrons (MLPs)
        *   Random Forest classifiers
    *   These models are trained on different variations of the dataset (raw images, pickled data).
    *   [More details](./MAHI_MODELS.md)

## Model Integration

All of these models are integrated into a single Graphical User Interface (GUI), which allows for easy comparison and testing. The `gui.py` script is responsible for loading the models and making predictions.

Each model has a specific set of functions for loading the model and for making predictions, which are called by the GUI. This modular approach allows for easy addition of new models in the future.
