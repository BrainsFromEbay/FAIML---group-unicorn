# Model Evaluation Results

This directory contains evaluation metrics and results for the trained models.

## Directory Structure

- **`cnn_mnist/`**
- **`mlp_mnist/`** 
- **`rf_mnist/`**
- **`cnn_pickle/`**
- **`mlp_pickle/`**
- **`rf_pickle/`**

## Summary of Results (Custom Test - Dec 19)

The following results reflect the models' performance on the **custom_test** dataset (20 images), which represents a realistic, out-of-distribution handwritten digit recognition task.

| Rank | Model Script | Accuracy | Correct/Total | Best at |
| :--: | :--- | :------: | :---: | :--- |
| **1** | **CNN Raw** | **85.0%** | 17/20 | Training on domain-specific data |
| **2** | **CNN MNIST** | **80.0%** | 16/20 | Generalizing (Standard Baseline) |
| **2** | **CNN Pickle** | **80.0%** | 16/20 | Handling "noisy" variants |
| 4 | MLP MNIST | 75.0% | 15/20 | Robust runner-up |
| 5 | MLP Pickle | 70.0% | 14/20 | Only model to identify '4' correctly |
| 6 | RF Inference | 45.0% | 9/20 | Weak Performance |
| 7 | RF MNIST | 40.0% | 8/20 | Weak Performance |

### Key Insights

1.  **CNN Supremacy**: Convolutional Neural Networks consistently outperform MLPs and Random Forests, regardless of the training data source.
2.  **Domain Specificity**: `CNN Raw`, trained on the raw dataset, achieved the highest accuracy (85%).
3.  **Preprocessing Alignment**: `CNN Pickle` performed exceptionally well on "noisy" images, suggesting its training data (pickle files) shares characteristics with these noisy artifacts.
4.  **Legacy Metrics**: Previous internal validation showed >98% accuracy for many models, which was likely due to overfitting or data leakage. The `custom_test` results above are a much more accurate reflection of real-world utility.

## Contents

Each folder contains `stats.md` and `confusion_matrix.png` generated from the *validation* phase (not the inference phase above).
