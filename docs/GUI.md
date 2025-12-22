# Graphical User Interface (GUI)

This project includes a user-friendly Graphical User Interface (GUI) built with [Streamlit](https://streamlit.io/) that allows you to interact with all the trained models in real-time.

## How to Run the GUI

To start the GUI, navigate to the root directory of the project in your terminal and run the following command:

```bash
streamlit run gui.py
```

This will open a new tab in your web browser with the application running.

## Features

The GUI is divided into several sections:

### 1. Model Selection

*   **Location:** Sidebar on the left.
*   **Functionality:** A radio button allows you to choose which of the trained models you want to use for prediction. The available models are:
    *   Mahi - CNN Raw
    *   Mahi - CNN Pickle
    *   Mahi - MLP MNIST
    *   Mahi - MLP Pickle
    *   Mahi - Random Forest
    *   Oyshe - HOG + Logistic Regression
    *   Jere - CNN

### 2. Input Image

*   **Location:** Main content area.
*   **Functionality:** You can provide an image of a handwritten digit in two ways:
    *   **Upload Image:** Upload a PNG or JPG file directly from your computer.
    *   **Select from Folder:** Enter the path to a folder (e.g., `custom_test`) and then select an image from a dropdown menu.

Once an image is selected or uploaded, it will be displayed on the screen.

### 3. Prediction

*   **Location:** Main content area, next to the input image.
*   **Functionality:** After selecting a model and providing an image, click the "Predict Digit" button. The application will then:
    1.  Load the selected model (models are cached for performance).
    2.  Preprocess the input image to match the format expected by the model.
    3.  Perform the prediction.
    4.  Display the predicted digit and the model's confidence score.

### 4. Model Performance & Statistics

*   **Location:** Bottom of the main content area.
*   **Functionality:** This section displays the confusion matrix for the currently selected model, providing a visual representation of its performance. The confusion matrices are pre-generated and stored in the `results` directory for each model.

### 5. Model Description

*   **Location:** Bottom of the main content area.
*   **Functionality:** This section provides a brief description of the selected model's architecture and training data.

## Integration of Models

The `gui.py` script imports the necessary prediction functions from each of the model's respective `inference` or `predict` scripts. When a model is selected in the GUI, the corresponding functions are called to load the model and perform the prediction. This modular design makes it easy to add new models to the GUI in the future.
