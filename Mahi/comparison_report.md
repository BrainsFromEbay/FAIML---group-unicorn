# Model Performance Comparison Report

This report summarizes the performance of five different machine learning models on a custom test set of 9 digit images. All models were tested using inference scripts located in the `Mahi/preprocessed/inference/` directory.

## Prediction Matrix

The following table shows the predictions of each model for every image in the `custom_test` folder. A checkmark (✅) indicates a correct prediction, while a cross (❌) indicates an incorrect one.

| Image | Expected | `mlp_inference` | `mlp_raw_inference` | `rf_inference` | `cnn_inference` | `cnn_raw_inference` |
| :---- | :------: | :-------------: | :-----------------: | :------------: | :-------------: | :-----------------: |
| 0.png | 0        | 1 ❌            | 1 ❌                | 1 ❌           | 8 ❌            | 8 ❌                |
| 1.png | 1        | 1 ✅            | 1 ✅                | 2 ❌           | 1 ✅            | 1 ✅                |
| 2.png | 2        | 2 ✅            | 3 ❌                | 2 ✅           | 2 ✅            | 2 ✅                |
| 3.png | 3        | 1 ❌            | 1 ❌                | 7 ❌           | 3 ✅            | 3 ✅                |
| 4.png | 4        | 8 ❌            | 3 ❌                | 2 ❌           | 7 ❌            | 1 ❌                |
| 5.png | 5        | 3 ❌            | 3 ❌                | 5 ✅           | 3 ❌            | 5 ✅                |
| 6.png | 6        | 4 ❌            | 9 ❌                | 1 ❌           | 4 ❌            | 4 ❌                |
| 7.png | 7        | 7 ✅            | 7 ✅                | 7 ✅           | 7 ✅            | 7 ✅                |
| 9.png | 9        | 1 ❌            | 1 ❌                | 1 ❌           | 9 ✅            | 9 ✅                |

---

## Accuracy Scoreboard

This table ranks the models from best to worst based on their accuracy on the test set.

| Rank | Script / Model | Correct | Total | Accuracy |
| :--: | :--- | :-----: | :---: | :------: |
| 1    | `cnn_raw_inference.py` (`CNN_from_raw_image.pth`) | 6       | 9     | **66.7%**  |
| 2    | `cnn_inference.py` (`CNN_digit_full.pth`) | 5       | 9     | **55.6%**  |
| 3    | `mlp_inference.py` (`best_mlp_model.pth`) | 3       | 9     | **33.3%**  |
| 4    | `rf_inference.py` (`rf_model.joblib`) | 3       | 9     | **33.3%**  |
| 5    | `mlp_raw_inference.py` (`MLP_from_raw_image.pth`) | 2       | 9     | **22.2%**  |

---

## Analysis

### Key Observations:
1.  **CNNs Outperform All Other Models:** The two Convolutional Neural Network (CNN) models (`cnn_raw_inference` and `cnn_inference`) were the clear winners, achieving the highest and second-highest accuracy. This is expected, as CNNs are architecturally designed for image recognition tasks and are generally superior to MLPs and traditional machine learning models like Random Forest for this purpose.

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

The **`CNN_from_raw_image.pth` model is the best performing model** on this custom test set, with an accuracy of 66.7%. The results clearly demonstrate the architectural advantage of CNNs for image classification tasks compared to both MLPs and Random Forest classifiers.