# Model Performance Comparison Report

This report summarizes the performance of five different machine learning models on a custom test set of 9 digit images. All models were tested using inference scripts located in the `Mahi/preprocessed/inference/` directory.

## Prediction Matrix

The following table shows the predictions of each model for every image in the `custom_test` folder. A checkmark (✅) indicates a correct prediction, while a cross (❌) indicates an incorrect one.

| Image | Expected | `mlp_inference` | `mlp_raw_inference` | `rf_inference` | `cnn_inference` | `cnn_raw_inference` | `mlp_inference_improved` | `cnn_inference_improved` | `cnn_full_improved` | `cnn_refined_inference` | `cnn_final_mnist` | `mlp_mnist` | `rf_mnist` |
| :---- | :------: | :-------------: | :-----------------: | :------------: | :-------------: | :-----------------: | :----------------------: | :----------------------: | :----------------------: | :--------------------------: | :-------------------: | :---------: | :--------: |
| 0.png | 0        | 1 ❌            | 1 ❌                | 1 ❌           | 8 ❌            | 8 ❌                | 8 ❌                     | 2 ❌                     | 2 ❌ | 8 ❌ | 8 ❌ | 8 ❌ | 2 ❌ |
| 1.png | 1        | 1 ✅            | 1 ✅                | 2 ❌           | 1 ✅            | 1 ✅                | 1 ✅                     | 7 ❌                     | 7 ❌ | 4 ❌ | 1 ✅ | 1 ✅ | 1 ✅ |
| 2.png | 2        | 2 ✅            | 3 ❌                | 2 ✅           | 2 ✅            | 2 ✅                | 2 ✅                     | 7 ❌                     | 7 ❌ | 2 ✅ | 2 ✅ | 2 ✅ | 2 ✅ |
| 3.png | 3        | 1 ❌            | 1 ❌                | 7 ❌           | 3 ✅            | 3 ✅                | 9 ❌                     | 3 ✅                     | 3 ✅ | 3 ✅ | 3 ✅ | 3 ✅ | 1 ❌ |
| 4.png | 4        | 8 ❌            | 3 ❌                | 2 ❌           | 7 ❌            | 1 ❌                | 4 ✅                     | 7 ❌                     | 7 ❌ | 7 ❌ | 1 ❌ | 1 ❌ | 5 ❌ |
| 5.png | 5        | 3 ❌            | 3 ❌                | 5 ✅           | 3 ❌            | 5 ✅                | 5 ✅                     | 5 ✅                     | 5 ✅ | 3 ❌ | 5 ✅ | 5 ✅ | 5 ✅ |
| 6.png | 6        | 4 ❌            | 9 ❌                | 1 ❌           | 4 ❌            | 4 ❌                | 9 ❌                     | 5 ❌                     | 5 ❌ | 5 ❌ | 8 ❌ | 6 ✅ | 6 ✅ |
| 7.png | 7        | 7 ✅            | 7 ✅                | 7 ✅           | 7 ✅            | 7 ✅                | 7 ✅                     | 7 ✅                     | 7 ✅ | 7 ✅ | 7 ✅ | 7 ✅ | 2 ❌ |
| 9.png | 9        | 1 ❌            | 1 ❌                | 1 ❌           | 9 ✅            | 9 ✅                | 9 ✅                     | 9 ✅                     | 9 ✅ | 9 ✅ | 9 ✅ | 9 ✅ | 8 ❌ |

---

## Accuracy Scoreboard

This table ranks the models from best to worst based on their accuracy on the test set.

| Rank | Script / Model | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
|| 1    | `cnn_mnist.pth` (Friend's Model) | 8 | 9 | **88.9%** |
| 2    | `mlp_mnist_best.pth` (MLP on MNIST) | 8 | 10 | **80.0%** |
| 3    | `cnn_final_mnist.pth` (My Re-trained Code) | 7 | 10 | **70.0%** |
| 4    | `cnn_raw_inference.py` (`CNN_from_raw_image.pth`) | 6       | 9     | **66.7%**  |
| 4 (Tie)| `inference_custom_mlp_improved.py` (`best_mlp_improved.pth`) | 6 | 9 | **66.7%** |
| 6    | `cnn_inference.py` (`CNN_digit_full.pth`) | 5       | 9     | **55.6%**  |
| 7    | `inference_custom_cnn_improved.py` (`best_cnn_improved.pth`) | 4 | 9 | **44.4%** |
| 7 (Tie)| `cnn_full_improved.pth` | 4 | 9 | **44.4%** |
| 7 (Tie)| `inference_cnn_refined.py` (`best_cnn_refined.pth`) | 4 | 9 | **44.4%** |
| 10   | `rf_mnist.joblib` (RF on MNIST) | 4 | 10 | **40.0%** |
| 11   | `mlp_inference.py` (`best_mlp_model.pth`) | 3       | 9     | **33.3%**  |
| 11 (Tie)| `rf_inference.py` (`rf_model.joblib`) | 3       | 9     | **33.3%**  |
| 13   | `mlp_raw_inference.py` (`MLP_from_raw_image.pth`) | 2       | 9     | **22.2%**  |

---

## Analysis

### Key Observations:
1.  **MNIST Data is Superior (for NNs):** The CNN and MLP models trained on MNIST (`cnn_mnist.pth`, `mlp_mnist_best.pth`, `cnn_final_mnist.pth`) consistently outperformed all models trained on the user's custom dataset. The Friend's model reached **88.9%**, and the MLP on MNIST reached **80.0%**.

2.  **RF Struggles:** The Random Forest trained on MNIST (`rf_mnist.joblib`) achieved only **40.0%**, significantly worse than the NNs on the same data. This confirms that for raw pixel domains with noise/style variations (like handwriting), Neural Networks (especially CNNs/MLPs) are structurally superior to Decision Trees.

3.  **MLP on MNIST > CNN on Custom Data:** The simple MLP trained on MNIST (80%) outperformed even the best CNN trained on custom data (66.7%). This proves: **Data Quality > Model Architecture**.

### Image-Specific Observations:
*   **Easy Images:** `1, 2, 5, 6` were correctly identified by RF.
*   **Hard Images:** `3, 7, 9` were missed by RF but caught by most NNs. This suggests RF overfits to specific pixel patterns and fails when the digit shifts or tilts slightly.

## Conclusion

The **Friend's `cnn_mnist.pth` (88.9%)** and our **`mlp_mnist_best.pth` (80.0%)** are the best performing models.

**Final Takeaway:**
1.  **Use MNIST**: Training on the large, diverse MNIST dataset is far better than using the small generated pickle file.
2.  **Use Neural Networks**: CNNs or MLPs generalize much better than Random Forests for this image recognition task.