Based on an inspection of `random_forrest.py`, this model faces different challenges than the Deep Learning models (CNN/MLP). Random Forests are robust but treat pixels as independent features, making them poor at capturing spatial relationships without help.

Here are the specific issues and instructions to improve the Random Forest model:

### 1. Address Overfitting (Hyperparameter Tuning)

* **Observation:** The script initializes the model with default parameters: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`.
* **Impact:** By default, sklearn trees grow until all leaves are pure (perfectly memorizing the training subset). This leads to massive overfitting on noisy image data.
* **Instruction:** Limit the complexity of individual trees using `min_samples_split` or `max_depth`.
* **Action:** Replace the simple initialization with a **RandomizedSearchCV** to find better parameters.
* **Code Snippet:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)
search.fit(X_train_flat, y_train)

model = search.best_estimator_
print(f"Best parameters: {search.best_params_}")

```





### 2. Reduce Dimensionality (Noise Reduction)

* **Observation:** The images are flattened into vectors of size 1024 (32x32).
* **Impact:** Many of these pixels (especially near corners/edges) are likely background noise. Random Forests struggle when the ratio of "noise features" to "informative features" is high.
* **Instruction:** Apply **Principal Component Analysis (PCA)** before training to compress the 1024 pixels into the most important features (e.g., top 50-100 components).
* **Code Snippet:**
```python
from sklearn.decomposition import PCA

# Keep 95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)
X_val_pca = pca.transform(X_val_flat)

# Train model on X_train_pca instead of X_train_flat
model.fit(X_train_pca, y_train)

```





### 3. Implement Offline Data Augmentation

* **Observation:** Like the other scripts, this trains on static images.
* **Impact:** The Random Forest learns that a "7" exists only at specific pixel coordinates.
* **Instruction:** Since sklearn doesn't have a `DataLoader` with on-the-fly transforms, you must generate augmented data *before* training.
* **Action:** Use `scipy.ndimage` or `cv2` to create rotated/shifted versions of your training set and append them to `X_train`.
* **Code Snippet:**
```python
from scipy.ndimage import rotate, shift

augmented_X = []
augmented_y = []

for img, label in zip(X_train, y_train):
    # Add original
    augmented_X.append(img.flatten())
    augmented_y.append(label)

    # Add rotated version (+10 degrees)
    # Reshape to 2D first, rotate, flatten back
    img_rot = rotate(img, angle=10, reshape=False)
    augmented_X.append(img_rot.flatten())
    augmented_y.append(label)

    # Add rotated version (-10 degrees)
    img_rot_neg = rotate(img, angle=-10, reshape=False)
    augmented_X.append(img_rot_neg.flatten())
    augmented_y.append(label)

X_train_aug = np.array(augmented_X)
y_train_aug = np.array(augmented_y)

# Now fit on X_train_aug

```





### Summary of Priority Actions for `random_forrest.py`:

1. **High:** Constrain the trees (e.g., `min_samples_leaf=4`) to prevent memorization.
2. **Medium:** Use PCA to reduce the feature space from 1024 pixels to ~50-80 components.
3. **Medium:** Manually expand the training dataset with rotated copies of the images.