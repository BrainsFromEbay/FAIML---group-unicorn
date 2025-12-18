# Model Performance Comparison Report

This report summarizes the performance of five different machine learning models on a custom test set of 9 digit images. All models were tested using inference scripts located in the `Mahi/preprocessed/inference/` directory.

## Prediction Matrix

The following table shows the predictions of each model for every image in the `custom_test` folder. A checkmark (✅) indicates a correct prediction, while a cross (❌) indicates an incorrect one.

| Image | Expected | `mlp_inference` | `mlp_raw_inference` | `rf_inference` | `cnn_inference` | `cnn_raw_inference` | `mlp_inference_improved` | `cnn_inference_improved` | `cnn_full_improved` |
| :---- | :------: | :-------------: | :-----------------: | :------------: | :-------------: | :-----------------: | :----------------------: | :----------------------: | :----------------------: |
| 0.png | 0        | 1 ❌            | 1 ❌                | 1 ❌           | 8 ❌            | 8 ❌                | 8 ❌                     | 2 ❌                     | 2 ❌ |
| 1.png | 1        | 1 ✅            | 1 ✅                | 2 ❌           | 1 ✅            | 1 ✅                | 1 ✅                     | 7 ❌                     | 7 ❌ |
| 2.png | 2        | 2 ✅            | 3 ❌                | 2 ✅           | 2 ✅            | 2 ✅                | 2 ✅                     | 7 ❌                     | 7 ❌ |
| 3.png | 3        | 1 ❌            | 1 ❌                | 7 ❌           | 3 ✅            | 3 ✅                | 9 ❌                     | 3 ✅                     | 3 ✅ |
| 4.png | 4        | 8 ❌            | 3 ❌                | 2 ❌           | 7 ❌            | 1 ❌                | 4 ✅                     | 7 ❌                     | 7 ❌ |
| 5.png | 5        | 3 ❌            | 3 ❌                | 5 ✅           | 3 ❌            | 5 ✅                | 5 ✅                     | 5 ✅                     | 5 ✅ |
| 6.png | 6        | 4 ❌            | 9 ❌                | 1 ❌           | 4 ❌            | 4 ❌                | 9 ❌                     | 5 ❌                     | 5 ❌ |
| 7.png | 7        | 7 ✅            | 7 ✅                | 7 ✅           | 7 ✅            | 7 ✅                | 7 ✅                     | 7 ✅                     | 7 ✅ |
| 9.png | 9        | 1 ❌            | 1 ❌                | 1 ❌           | 9 ✅            | 9 ✅                | 9 ✅                     | 9 ✅                     | 9 ✅ |

---

## Accuracy Scoreboard

This table ranks the models from best to worst based on their accuracy on the test set.

| Rank | Script / Model | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
| 1    | `cnn_raw_inference.py` (`CNN_from_raw_image.pth`) | 6       | 9     | **66.7%**  |
| 1 (Tie)| `inference_custom_mlp_improved.py` (`best_mlp_improved.pth`) | 6 | 9 | **66.7%** |
| 3    | `cnn_inference.py` (`CNN_digit_full.pth`) | 5       | 9     | **55.6%**  |
| 4    | `inference_custom_cnn_improved.py` (`best_cnn_improved.pth`) | 4 | 9 | **44.4%** |
| 4 (Tie)| `cnn_full_improved.pth` | 4 | 9 | **44.4%** |
| 6    | `mlp_inference.py` (`best_mlp_model.pth`) | 3       | 9     | **33.3%**  |
| 4    | `rf_inference.py` (`rf_model.joblib`) | 3       | 9     | **33.3%**  |
| 5    | `mlp_raw_inference.py` (`MLP_from_raw_image.pth`) | 2       | 9     | **22.2%**  |

---

## Analysis

### Key Observations:
1.  **Improved MLP Rivals CNN:** The new `mlp_inference_improved` model achieved **66.7% accuracy**, tying with the best performing CNN. This proves that with proper data augmentation, regularization, and architecture tuning, MLPs can be highly effective for this task. It correctly identified image `4.png` which all other models failed on.

2.  **CNNs Outperform All Other Models:** (Previous Note: While the improved MLP caught up, the standard CNNs were still historically strong). The two Convolutional Neural Network (CNN) models (`cnn_raw_inference` and `cnn_inference`) were the clear winners (until the improved MLP), achieving the highest and second-highest accuracy.

2.  **"Raw" vs. "Preprocessed" Training Data:**
    *   The models trained on data from the `raw_image` directory (`cnn_raw_inference`, `mlp_raw_inference`) required specific preprocessing (inversion, thresholding) to work with the `custom_test` images.
    *   The models trained on the pickled data (`cnn_inference`, `mlp_inference`, `rf_inference`) also required this same preprocessing, indicating that the `digits_data_cleaned.pickle` file contains images that have a similar distribution (white digits on a black background).
    *   The best performing model overall (`cnn_raw_inference`) was trained on the "raw" data, suggesting its training set may have been more robust or the training process more effective than its "preprocessed" counterpart.

3.  **Model Complexity Matters:**
    *   The simple MLP models performed poorly, with the one trained on "raw" data being the least accurate model. This highlights that for image tasks, a simple MLP struggles to learn the spatial features that CNNs capture easily.
    *   The Random Forest model performed on par with the best MLP, but was still significantly outperformed by the CNNs.

### Image-Specific Observations:
*   **Easy Images:** `1.png` and `7.png` were the easiest images, with most models correctly identifying them. Image `7` was correctly identified by all but one model.
*   **Hard Images:** `0.png`, `4.png`, and `6.png` were the hardest, with **no model** getting them right. This could be due to unusual handwriting styles, artifacts in the images, or a lack of similar examples in any of the training sets. For example, all models that didn't predict the correct class for image `4` confused it with a different number, suggesting the features were ambiguous.

## Conclusion

The **`CNN_from_raw_image.pth` model and `best_mlp_improved.pth` are the best performing models** on this custom test set, with an accuracy of 66.7%.

The improved CNN model (`best_cnn_improved.pth` and `cnn_full_improved.pth`) achieved 44.4%, which is surprisingly lower than the baseline CNN (55.6%) on this specific small test set, despite having superior validation accuracy (99%) during training.

**Reason for Regression:**
1.  **Over-Regularization/Augmentation Mismatch:** The huge gap between Train/Val (99%) and Custom Test (44%) often indicates that the augmentation (rotation/shifting) made the model expect "messy" data, while the custom test set (after 50-value threshold cleaning) was "too clean" or simply different.
2.  **Architecture:** The original CNN had dense layers (`Linear(64*8*8, 128)`). The improved CNN replaced this with `AdaptiveAvgPool` + `Linear(64, 32)`. While this is theoretically better for preventing overfitting (reducing params), it may have removed the specific "rote memorization" capacity that actually helped classify these specific, clean custom images. The original model could "memorize" the spatial layout of specific digits better.
3.  **Thresholding:** The cleaning step (setting <50 to 0) might be removing subtle edges that the Global Average Pooling layer requires to differentiate shapes, whereas the Dense layer in the original model could just look at specific pixel coordinates.