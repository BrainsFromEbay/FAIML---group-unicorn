# Oyshe's HOG + Logistic Regression Model

This document details the machine learning model implemented by Oyshe, which uses the Histogram of Oriented Gradients (HOG) feature descriptor combined with a Logistic Regression classifier.

## Overview

Oyshe's model represents a classical computer vision approach to the digit recognition problem. Instead of feeding raw pixel data into a neural network, it first extracts meaningful features from the images and then uses a simpler classifier to make predictions.

The code for this model is located in the `Oyshe/` directory.

## Feature Extraction: HOG

The core of this model is the use of Histogram of Oriented Gradients (HOG) features. HOG is a feature descriptor used in computer vision and image processing for the purpose of object detection.

*   **How it works:** The HOG descriptor technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.

*   **Implementation:** The HOG features are extracted using the `skimage.feature.hog` function from the scikit-image library. The parameters used for the HOG extraction are defined in `Oyshe/prediction.py`.

## Model: Logistic Regression

After extracting the HOG features, a simple and efficient **Logistic Regression** classifier is used to predict the digit. Logistic Regression is a linear model that can be used for classification.

## Training

The training process is handled by the `Oyshe/model.py` script. The script performs the following steps:

1.  **Loads the MNIST dataset:** It manually parses the MNIST IDX format files.
2.  **Extracts HOG features:** For each image in the training set, it computes the HOG features.
3.  **Trains the Logistic Regression model:** It uses a custom-implemented Logistic Regression model to learn the relationship between the HOG features and the digit labels.
4.  **Saves the model:** The trained weights and bias of the logistic regression model are saved to a NumPy NPZ file: `Oyshe/hog_logistic_mnist.npz`.

To train the model, you can run the `model.py` script:
```bash
python Oyshe/model.py
```

## Prediction

The `Oyshe/prediction.py` script is responsible for making predictions with the trained model. It includes the following key functions:

*   `load_model()`: Loads the weights and bias from `Oyshe/hog_logistic_mnist.npz`.
*   `preprocess_image(image_path)`: Preprocesses an input image by resizing it to 28x28, converting to grayscale, inverting colors if necessary, and then extracting the HOG features in the same way as during training.
*   `predict_digit(image_path, weights, bias)`: Takes the preprocessed HOG features of an image and uses the loaded model (weights and bias) to predict the digit and confidence.

These functions are imported and used by the main `gui.py` to integrate Oyshe's model into the application.
