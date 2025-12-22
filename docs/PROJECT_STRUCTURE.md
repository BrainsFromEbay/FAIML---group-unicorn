# Project Structure

This project is organized into several directories and files, each with a specific purpose. The following is a breakdown of the project's structure:

```
├───.gitignore
├───environment.yml
├───gui.py
├───README.md
├───requirements.txt
├───custom_test/
├───data/
│   └───MNIST/
├───Jere/
│   ├───cnn_mnist.pth
│   ├───cnn_mnist.py
│   ├───predict.py
│   └───README.md
├───Mahi/
│   ├───src/
│   │   ├───data_preprocessing/
│   │   ├───inference/
│   │   ├───models/
│   │   └───train_script/
│   └───results/
├───Oyshe/
│   ├───hog_logistic_mnist.npz
│   ├───model.py
│   ├───prediction.py
│   └───README.md
└───docs/
    ├───README.md
    ├───PROJECT_STRUCTURE.md
    ├───SETUP.md
    ├───DATA.md
    ├───MODELS.md
    ├───JERE_CNN.md
    ├───OYSHE_HOG.md
    ├───MAHI_MODELS.md
    ├───GUI.md
    └───RESULTS.md
```

## Top-Level Files

*   `gui.py`: The main graphical user interface for the project, built with Streamlit. It integrates all the models and allows for real-time digit prediction.
*   `requirements.txt`: A list of Python packages required to run the project.
*   `environment.yml`: A Conda environment file that specifies the dependencies for the project.
*   `README.md`: The main README file for the project.

## Directories

*   `docs/`: Contains all the documentation for the project.
*   `custom_test/`: A directory containing custom-drawn digit images (0-9) used for testing the models' performance on data not from the MNIST dataset.
*   `data/`: Intended to store datasets. The MNIST dataset is expected to be in `data/MNIST/`.
*   `Jere/`: Contains the work of Jere, which is a Convolutional Neural Network (CNN) model for MNIST.
    *   `cnn_mnist.py`: The script to train the CNN model.
    *   `predict.py`: The script to make predictions using the trained CNN model.
    *   `cnn_mnist.pth`: The saved weights of the trained CNN model.
*   `Mahi/`: Contains the work of Mahi, which includes a variety of models like CNN, MLP, and Random Forest. This directory is further divided into `src` for the source code and `results` for the model evaluation results.
    *   `src/data_preprocessing/`: Scripts for processing the data.
    *   `src/train_script/`: Scripts for training the different models.
    *   `src/inference/`: Scripts for making predictions with the trained models.
    *   `src/models/`: Saved model files.
    *   `results/`: Stores confusion matrices and statistics for the models.
*   `Oyshe/`: Contains the work of Oyshe, which is a Logistic Regression model using Histogram of Oriented Gradients (HOG) features.
    *   `model.py`: The script to train the HOG-based Logistic Regression model.
    *   `prediction.py`: The script to make predictions using the trained model.
    *   `hog_logistic_mnist.npz`: The saved model parameters.
