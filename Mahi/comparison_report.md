# Model Performance Comparison Report

This report summarizes the performance of the active models in `Mahi/preprocessed/inference/`.
All models were re-evaluated on the `custom_test` dataset (10 images).

## Prediction Matrix

| Image | Exp | CNN Raw | CNN Pickle | CNN MNIST | MLP Pickle | MLP MNIST | RF Pickle | RF MNIST |
| :---: | :-: | :-----: | :--------: | :-------: | :--------: | :-------: | :-------: | :------: |
| 0.png |  0  |  8 ❌   |    8 ❌    |   8 ❌    |    8 ❌    |   8 ❌    |   1 ❌    |   2 ❌   |
| 1.png |  1  |  1 ✅   |    1 ✅    |   1 ✅    |    1 ✅    |   1 ✅    |   2 ❌    |   1 ✅   |
| 2.png |  2  |  2 ✅   |    7 ❌    |   2 ✅    |    2 ✅    |   2 ✅    |   2 ✅    |   2 ✅   |
| 3.png |  3  |  3 ✅   |    3 ✅    |   3 ✅    |    9 ❌    |   3 ✅    |   7 ❌    |   1 ❌   |
| 4.png |  4  |  1 ❌   |    7 ❌    |   1 ❌    |    4 ✅    |   1 ❌    |   2 ❌    |   5 ❌   |
| 5.png |  5  |  5 ✅   |    3 ❌    |   5 ✅    |    5 ✅    |   1 ❌    |   5 ✅    |   5 ✅   |
| 6.png |  6  |  4 ❌   |    0 ❌    |   8 ❌    |    9 ❌    |   6 ✅    |   1 ❌    |   6 ✅   |
| 7.png |  7  |  7 ✅   |    7 ✅    |   7 ✅    |    7 ✅    |   7 ✅    |   7 ✅    |   2 ❌   |
| 8.png |  8  |  8 ✅   |    8 ✅    |   8 ✅    |    8 ✅    |   8 ✅    |   1 ❌    |   2 ❌   |
| 9.png |  9  |  9 ✅   |    9 ✅    |   9 ✅    |    9 ✅    |   9 ✅    |   1 ❌    |   8 ❌   |

> **Note**: `CNN Raw` corresponds to `cnn_raw_inference.py`, `CNN MNIST` to `cnn_mnist_inference.py`, etc.

## Accuracy Scoreboard

| Rank | Model Script | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
| 1 (Tie) | `cnn_raw_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `cnn_mnist_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `mlp_mnist_inference.py` | 7 | 10 | **70.0%** |
| 1 (Tie) | `mlp_pickle_inference.py` | 7 | 10 | **70.0%** |
| 5 | `cnn_pickle_inference.py` | 5 | 10 | **50.0%** |
| 6 | `rf_mnist_inference.py` | 4 | 10 | **40.0%** |
| 7 | `rf_inference.py` | 3 | 10 | **30.0%** |

## Analysis

1.  **Neural Networks > Random Forest**: The neural networks (CNNs and MLPs) consistently outperformed Random Forests (30-40%). This confirms that for image data, especially with domain shifts (custom handwriting), deep learning approaches are far more robust.
2.  **Consistency**: Four different NN models achieved **70% accuracy**. This suggests a "ceiling" at 70% for these standard architectures on this specific test set without more advanced techniques or more targeted data.
    - They mostly fail on the same difficult digits: **0** (looks like 8), **4** (confusing stroke), and sometimes **6**.
3.  **Surprise Performer**: `MLP Pickle` was the **only model to correctly classify the digit 4**. This is notable because even the MNIST models failed on it.
4.  **MNIST Benefit**: While `RF` improved on MNIST (30% -> 40%) and `MLP` maintained high performance (70%), the `CNN` didn't dramatically outperform `CNN Raw` in this specific run. However, historically ( फ्रेंड's model), MNIST has shown potential for higher accuracy (89%).

## Conclusion

The project has consolidated around 70% accuracy for Neural Networks. To break this ceiling, one would likely need to:
1.  Augment the training data with samples that specifically resemble the fail cases (0s that look like 8s, 4s with this specific stroke style).
2.  Use a more complex architecture or ensemble methods.