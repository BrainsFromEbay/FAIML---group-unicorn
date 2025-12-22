# Jere's CNN Model

This document provides a detailed description of the Convolutional Neural Network (CNN) model implemented by Jere for handwritten digit recognition.

## Overview

Jere's contribution to the project is a straightforward and effective CNN model built using the PyTorch library. The model is designed to be trained on the MNIST dataset and is capable of achieving high accuracy in digit classification.

The code for this model is located in the `Jere/` directory.

## Model Architecture

The CNN architecture is defined in the `Jere/cnn_mnist.py` script, within the `SimpleCNN` class. It consists of two convolutional layers followed by two fully connected (linear) layers.

The architecture is as follows:

1.  **Convolutional Layer 1:**
    *   1 input channel, 16 output channels
    *   3x3 kernel size
    *   ReLU activation
    *   2x2 Max Pooling

2.  **Convolutional Layer 2:**
    *   16 input channels, 32 output channels
    *   3x3 kernel size
    *   ReLU activation
    *   2x2 Max Pooling

3.  **Fully Connected Layer 1:**
    *   Input features are flattened from the previous layer.
    *   128 output features
    *   ReLU activation

4.  **Fully Connected Layer 2 (Output Layer):**
    *   10 output features, corresponding to the 10 digits (0-9).

## Training

The training process is also handled by the `Jere/cnn_mnist.py` script. The script performs the following steps:

1.  **Loads the MNIST dataset:** It uses `torchvision.datasets.MNIST` to download and load the training and testing data.
2.  **Initializes the model:** It creates an instance of the `SimpleCNN` model.
3.  **Defines the loss function and optimizer:** It uses Cross-Entropy Loss as the loss function and the Adam optimizer.
4.  **Trains the model:** It iterates through the training data for a specified number of epochs, updating the model's weights.
5.  **Saves the trained model:** After training, the model's state dictionary is saved to `Jere/cnn_mnist.pth`.

To train the model, you can run the `cnn_mnist.py` script:
```bash
python Jere/cnn_mnist.py
```

## Prediction

The `Jere/predict.py` script provides the functionality to use the trained model for predictions. It contains three main functions:

*   `load_model()`: Loads the saved model from `Jere/cnn_mnist.pth`.
*   `preprocess_image(image_path)`: Takes an image path as input, converts the image to grayscale, resizes it to 28x28, inverts the colors (to match the MNIST format), and converts it to a PyTorch tensor.
*   `predict_image(model, image_path)`: Takes a loaded model and an image path, preprocesses the image, and returns the predicted digit and the model's confidence.

These functions are imported and used by the main `gui.py` to integrate Jere's model into the application.
