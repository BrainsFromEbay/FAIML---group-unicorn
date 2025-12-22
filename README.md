# FAIML Project -- Group Unicorn

## Project Overview

This project explores various machine learning approaches for handwritten digit recognition, specifically targeting english digits from 0 to 9. We have explored different architectures (CNNs, MLPs, Random Forests, Logistic Regression) and training data sources (Raw Images, Pickled Data, MNIST).

## Team Contributions

The project is organized by team member:

- **[Mahi](Mahi/README.md)**: Explored a wide range of models (CNN, MLP, Random Forest) trained on an open data source found in kaggle [(see the datasethere)](https://www.kaggle.com/datasets/vaibhao/handwritten-characters). Followed two different approaches to train the models, one using raw images and the other using binary pickled data by preprocessing the raw images.
- **[Jere](Jere/README.md)**: Implemented a SimpleCNN trained on the official MNIST dataset available in pytorch. This is the best performing model in terms of generalization.
- **[Oyshe](Oyshe/README.md)**: Implemented a classic Computer Vision approach using HOG (Histogram of Oriented Gradients) feature extraction coupled with Logistic Regression. This is model is also trained on the MNIST dataset.

---

## 🚀 Quick Start (GUI)

We have built a **Streamlit GUI** (`gui.py`) that unifies all the models into a single interface. You can upload an image or select a folder to test predictions in real-time.

### 📦 Installation

First of all please download rf_mnist.joblib file from this [link](https://drive.google.com/file/d/1bevNLagSOjP-m3auP7qSJ6TyaqocUBla/view?usp=sharing) and place it in this location "Mahi/src/models/rf_mnist.joblib".

Additionaly you can download the pickle data from this [link](https://drive.google.com/file/d/1JJ9WHbLCZDNVymFtcXM1cAB6A6tGdMk6/view?usp=sharing) and place in "Mahi/src/data_preprocessing/digits_data.pickle" and run the script "Mahi/src/data_preprocessing/pickle_clean.py" to generate digits_data_cleaned.pickle file. This file is used for training the models.

To replicate the environment, you can install the dependencies using `pip` or `conda`.

### Using pip
```bash
pip install -r requirements.txt
```

### Using Conda
```bash
conda env create -f environment.yml
conda activate thesis
```

1.  **Run the App**:
    ```bash
    streamlit run gui.py
    ```
2.  **Features**:
    *   **Model Selection**: Choose from a wide variety of models:
        *   **Mahi**: CNN Raw, CNN Pickle, MLP MNIST, MLP Pickle, Random Forest.
        *   **Oyshe**: HOG + Logistic Regression.
        *   **Jere**: Simple CNN.
    *   **Input**: Upload an image or point to a local folder. You can type relative path to the folder if its inside the project directory. The path should be relative to the project root directory. If the folder is outside of the project directory, you can type the absolute path to the folder. (e.g., `custom_test`).
    *   **Results**: View real-time predictions with confidence scores.
    *   **Model Description**: See detailed architecture and training info for the selected model.
    *   **Performance**: View pre-calculated confusion matrices for *every* model.



---


# Unified Model Performance Leaderboard

We have tested all of the models on the same `custom_test` dataset (20 images). These images are outside of the training and validation datasets. We custom made these images. 

| Rank | Team Member | Model Variant | Architecture | Training Data | Accuracy |
| :--: | :---------- | :------------ | :----------- | :------------ | :------- |
| **1** | **Mahi** | **CNN Raw** | **CNN** | **Raw Images** | **85.0%** |
| **1** | **Jere** | **SimpleCNN** | **CNN** | **MNIST** | **85.0%** |
| 3 | Mahi | CNN Pickle | CNN | Pickled Data | 80.0% |
| 4 | Mahi | MLP MNIST | MLP | MNIST | 75.0% |
| 5 | Mahi | MLP Pickle | MLP | Pickled Data | 70.0% |
| 6 | Oyshe | HOG + LogReg | LogReg | MNIST | 60.0% |
| 7 | Mahi | Random Forest | RF | Pickled Data | 45.0% |
| 8 | Mahi | Random Forest | RF | MNIST | 40.0% |



## Detailed Reports

For more details, please refer to the individual directories:

- [Mahi/README.md](Mahi/README.md)
- [Jere/README.md](Jere/README.md)
- [Oyshe/README.md](Oyshe/README.md)
