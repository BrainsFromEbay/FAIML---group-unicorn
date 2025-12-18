The primary reason your models are likely overfitting (high training accuracy, lower validation accuracy) is the **lack of regularization** and **absence of data augmentation**.

### 1. Implement Data Augmentation (Highest Priority)

Your current pipelines feed the exact same pixel values to the model every epoch. The model "memorizes" these specific pixels instead of learning general shapes.

* **For `_from_raw_image.py` scripts:**
Update your `transforms.Compose` to include random variations.
* **Current:**
```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

```


* **Change to:**
```python
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),      # Rotate +/- 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Shift image slightly
    transforms.ToTensor(),
])
# Keep validation transform simple (just Resize and ToTensor)

```




* **For `_from_pickle.py` scripts:**
You currently manually process tensors in `__getitem__` without transforms. You should wrap the numpy arrays in `PIL` images or use `kornia` / `torchvision.transforms` on the tensors directly within the `__getitem__` method to apply similar random rotations and shifts.

### 2. Add Normalization

Your models currently scale data to `[0, 1]` (dividing by 255) but do not center it (mean=0, std=1). Standardizing inputs helps gradients flow better.

* **Action:** Calculate the mean and standard deviation of your training dataset and add `transforms.Normalize((mean,), (std,))` to your transform pipeline.
* **Code update:**
```python
transforms.Normalize((0.5,), (0.5,)) # Approximate values if exact mean/std are unknown

```



### 3. Add Regularization to Architectures

Some of your models are missing standard regularization layers that prevent overfitting.

* **Fix `CNN_from_raw_image.py`:**
The `SimpleCNN` classifier has **no Dropout**.
* **Instruction:** Add `nn.Dropout(0.5)` before the final fully connected layer.
* **Instruction:** Add `nn.BatchNorm2d` layers after every `nn.Conv2d` layer in `self.features`.


* **Fix `MLP_from_raw_image.py`:**
This model is purely linear layers with ReLU, with **zero Dropout**.
* **Instruction:** Add `nn.Dropout(0.2)` or `0.3` after every `nn.ReLU()` in the `self.network` sequence.


* **Note on `CNN_from_pickle.py`:**
This model *does* have Dropout (`0.5` and `0.3`). If it still overfits, increase the dropout rate or reduce the model size (see point 4).

### 4. Reduce Model Complexity (Parameter Count)

In `CNN_from_pickle.py`, the transition from Convolutional layers to Linear layers creates a massive number of parameters because the feature map is flattened directly.

* **Current:** `nn.Linear(64 * 8 * 8, 128)` creates ~524k parameters.
* **Instruction:** Add a **Global Average Pooling** layer before the classifier. This averages the 8x8 feature map into 1x1, drastically reducing parameters and forcing the model to learn robust features rather than specific pixel locations.
* *Change:*
```python
# In self.features
nn.AdaptiveAvgPool2d((1, 1))

# In self.classifier
nn.Linear(64, 128) # Input becomes 64 instead of 4096

```





### 5. Training Process Improvements

* **Weight Decay:** In all your `optim.Adam` calls, the `weight_decay` parameter is default (0).
* **Instruction:** Set `weight_decay=1e-4` or `1e-5`. This adds L2 regularization, penalizing large weights.
* `optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)`


* **Learning Rate Scheduler:** You are using a constant learning rate.
* **Instruction:** Use `torch.optim.lr_scheduler.ReduceLROnPlateau` to lower the learning rate when validation accuracy stops improving. This helps the model settle into a better minimum.