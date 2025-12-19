# Oyshe Model Results

This directory contains evaluation metrics for the HOG + Logistic Regression model.

## Directory Structure
- **`hog_logreg/`**: Contains `stats.md` and `confusion_matrix.png` generated from the `custom_test` dataset.

## Summary (Custom Test)

| Model | Accuracy | Feature Extraction | Classifier |
| :--- | :---: | :--- | :--- |
| **HOG + Logistic Regression** | **60.0%** | Histogram of Oriented Gradients (HOG) | Logistic Regression (Softmax) |

## Key Insights
- **Performance**: 12/20 correct (60%).
- **Strengths**: Robust on '0', '1', '2', '5', '7'.
- **Weaknesses**: Confuses '4', '6', '8', '3' often with '9'.
- **Comparison**: Outperforms random guessing (10%) and even some un-tuned CNNs (e.g. `RF Pickle` at 45%), showing the power of hand-crafted features for specific domains.
