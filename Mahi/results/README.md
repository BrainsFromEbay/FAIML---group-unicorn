# Model Evaluation Results

This directory contains evaluation metrics.

## Directory Structure

- **`cnn_mnist/`**
- **`mlp_mnist/`** 
- **`rf_mnist/`**
- **`cnn_pickle/`**
- **`mlp_pickle/`**
- **`rf_pickle/`**

## Summary of Results

| Model | Evaluation Dataset | Accuracy |
| :--- | :--- | :--- |
| **CNN MNIST** | MNIST Test Set (10k images) | **99%** |
| **MLP MNIST** | MNIST Test Set (10k images) | **98%** |
| **RF MNIST** | MNIST Test Set (10k images) | **97%** |
| **CNN Pickle** | Custom Val Split (11k images) | **100%** (Likely Overfit/Leakage) |
| **MLP Pickle** | Custom Val Split (11k images) | **100%** (Likely Overfit/Leakage) |
| **RF Pickle** | Custom Val Split (11k images) | **100%** (Likely Overfit/Leakage) |

> **Note**: The models trained on `digits_data_cleaned.pickle` show 100% accuracy on their validation set. This strongly suggests that the validation split in the pickle file contains duplicates of the training data. The **MNIST-trained models** provide a more realistic assessment of generalization performance (98-99%).

## Contents

Each folder contains `stats.md` and `confusion_matrix.png`.
