# Jere

## Dataset

The dataset used for training is the **MNIST** dataset.
The project uses the `torchvision.datasets.MNIST` class.

## Setup and Installation

**Dependencies:**
- `torch`
- `torchvision`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `seaborn`

To train the model:
```bash
python Jere/cnn_mnist.py
```
This generates `cnn_mnist.pth`.

To run prediction and generate results:
```bash
python Jere/predict.py
```

## File Descriptions

- `README.md`: This file.
- `cnn_mnist.py`: Training script for SimpleCNN.
- `predict.py`: Inference script for `custom_test` dataset.
- `results/`: Directory containing evaluation artifacts.

---

# Model Performance Comparison Report

The model was evaluated on the `custom_test` dataset (20 images).

## Prediction Matrix (Full Test Set)

| Image | Expected | Predicted | Confidence | Result |
| :---: | :------: | :-------: | :--------: | :----: |
| 0(1).png | 0 | 0 | 100.0% | ✅ |
| 0.png | 0 | 0 | 99.6% | ✅ |
| 1(1).png | 1 | 1 | 99.8% | ✅ |
| 1.png | 1 | 1 | 87.5% | ✅ |
| 2(1).png | 2 | 2 | 100.0% | ✅ |
| 2.png | 2 | 2 | 89.7% | ✅ |
| 3(1).png | 3 | 3 | 100.0% | ✅ |
| 3.png | 3 | 3 | 99.8% | ✅ |
| 4(1).png | 4 | 7 | 95.9% | ❌ |
| 4.png | 4 | 1 | 34.8% | ❌ |
| 5(1).png | 5 | 5 | 100.0% | ✅ |
| 5.png | 5 | 5 | 57.3% | ✅ |
| 6(1).png | 6 | 6 | 100.0% | ✅ |
| 6.png | 6 | 6 | 86.0% | ✅ |
| 7(1).png | 7 | 7 | 97.0% | ✅ |
| 7.png | 7 | 7 | 99.6% | ✅ |
| 8(1).png | 8 | 8 | 100.0% | ✅ |
| 8.png | 8 | 8 | 98.9% | ✅ |
| 9(1).png | 9 | 3 | 49.9% | ❌ |
| 9.png | 9 | 9 | 99.3% | ✅ |

## Accuracy Scoreboard

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **85.0%** (17/20) |

## Analysis

1.  **Performance**: The model achieved **85% accuracy**, which is a strong result competing with the best models in the project (e.g. Mahi's CNN Raw).
2.  **Observations**: 
    - The model consistently failed on digit **4**. It misclassified the noisy '4(1)' as '7' with high confidence, and the clean '4' as '1'.
    - It also missed the noisy '9(1)', confusing it with '3'.
    - However, it showed perfect or near-perfect confidence on most other digits.
