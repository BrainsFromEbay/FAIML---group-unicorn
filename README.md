# FAIML Project -- group-unicorn

## Project Overview

This project explores various machine learning approaches for handwritten digit recognition, specifically targeting a custom dataset (`custom_test`) containing both clean and noisy images. The team explored different architectures (CNNs, MLPs, Random Forests, Logistic Regression) and training data sources (Raw Images, Pickled Data, MNIST).

## Team Contributions

The project is organized by team member:

- **[Mahi](Mahi/README.md)**: Explored a wide range of models (CNN, MLP, Random Forest) trained on three different data sources (Raw, Pickle, MNIST). Established the comprehensive benchmarking standards.
- **[Jere](Jere/README.md)**: Implemented a SimpleCNN trained on the official MNIST dataset, bypassing download restrictions by using local data.
- **[Oyshe](Oyshe/README.md)**: Implemented a classic Computer Vision approach using HOG (Histogram of Oriented Gradients) feature extraction coupled with Logistic Regression.


---

## 🚀 Quick Start (GUI)

We have built a **Streamlit GUI** (`gui.py`) that unifies all the models into a single interface. You can upload an image or select a folder to test predictions in real-time.

1.  **Run the App**:
    ```bash
    streamlit run gui.py
    ```
2.  **Features**:
    *   **Model Selection**: Choose between Mahi (CNN/MLP/RF), Oyshe (HOG+LR), and Jere (CNN).
    *   **Input**: Upload an image or point to a local folder (e.g., `custom_test`).
    *   **Metrics**: View real-time predictions and pre-calculated confusion matrices.

## 📦 Installation

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

---


# Unified Model Performance Leaderboard

All models were evaluated on the same `custom_test` dataset (20 images).

| Rank | Team Member | Model Architecture | Training Data | Accuracy |
| :--: | :---------- | :----------------- | :------------ | :------- |
| **1** | **Mahi** | **CNN** | **Raw Images** | **85.0%** |
| **1** | **Jere** | **SimpleCNN** | **MNIST** | **85.0%** |
| 3 | Mahi | CNN | MNIST | 80.0% |
| 3 | Mahi | CNN | Pickled Data | 80.0% |
| 5 | Mahi | MLP | MNIST | 75.0% |
| 6 | Mahi | MLP | Pickled Data | 70.0% |
| 7 | Oyshe | HOG + LogReg | MNIST | 60.0% |
| 8 | Mahi | Random Forest | Pickled Data | 45.0% |
| 9 | Mahi | Random Forest | MNIST | 40.0% |

## Consolidated Analysis

1.  **CNN Supremacy**: Convolutional Neural Networks (CNNs) consistently outperformed other architectures (MLP, RF, LogReg), occupying the top 4 spots with accuracies between 80-85%.
2.  **Data Source Impact**:
    - **Raw Images**: Surprisingly, Mahi's CNN trained on raw images tied for first place (85%), suggesting the raw data distribution closely matches the `custom_test` set.
    - **MNIST**: Models trained on MNIST (Jere's SimpleCNN, Mahi's CNN MNIST) were very robust (80-85%), proving that standard MNIST is a great proxy for this task.
    - **Feature Engineering**: Oyshe's HOG + Logistic Regression (60%) outperformed Random Forests (40-45%), showing that manual feature extraction can better capture shape properties than raw pixel-based tree methods when deep learning isn't used.
3.  **The "Digit 4" Problem**: A recurring theme across almost all models (except Mahi's MLP Pickle) was the difficulty in correctly classifying the digit '4', often mistaking it for '1' or '7'.
4.  **Noise Robustness**: Several models (especially those trained on Pickled data or using HOG) showed surprisingly good performance on "noisy" images (ending in `(1).png`), sometimes even better than on clean images.

## detailed Reports

For detailed per-image prediction matrices, confusion matrices, and code, please refer to the individual directories:

- [Mahi/README.md](Mahi/README.md)
- [Jere/README.md](Jere/README.md)
- [Oyshe/README.md](Oyshe/README.md)
