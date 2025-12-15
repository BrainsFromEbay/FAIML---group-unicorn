# Jere Villman

# Tasnuba Oyshe

# Mahi Talukder:

## Setup and Installati

This project uses Python and several dependencies. You can set up the environment using either `pip` with `requirements.txt` or `conda` with `environment.yml`.

### Using Conda (Recommended)

To create a Conda environment with all the necessary dependencies, run the following command from the project's root directory:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment named `thesis`. To activate it, use:

```bash
conda activate thesis
```

### Using Pip

If you are not using Conda, you can install the required packages using pip. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Handwritten Digit Recognition CNN Model

This project includes a Convolutional Neural Network (CNN) model for recognizing handwritten digits (0-9), trained on PNG images.

### Files

- **Training Script:** `Mahi/digits_CNN_from_png.py`
- **Trained Model:** `Mahi/digits_model_CNN__from_png.pth`

### Model Details

The model is a `SimpleCNN` implemented in PyTorch. The architecture is defined in the training script.

- **Input:** The model expects a 4D tensor of shape `(N, 1, 28, 28)`, where:
  - `N` is the number of images in the batch.
  - `1` represents the single channel (grayscale).
  - `28, 28` is the image dimension.
- **Output:** The model outputs a tensor of shape `(N, 10)`, containing the raw, unnormalized scores (logits) for each of the 10 classes.
- **Classes:** The classes are the digits '0' through '9', sorted. `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`

### Preparing an Image for the Model

To test the model on a new image, it must be preprocessed to match the format used during training.

1.  **Load the image:** Load the image in grayscale.
2.  **Resize:** Resize the image to 28x28 pixels.
3.  **Convert to Tensor:** Convert the image to a PyTorch tensor. The pixel values should be scaled to the range `[0.0, 1.0]`.
4.  **Add Batch Dimension:** Add a batch dimension to the tensor, changing its shape from `(1, 28, 28)` to `(1, 1, 28, 28)`.

### Example Helper Function

Here is a Python code snippet demonstrating how to load the model and create a helper function to predict the digit from an image file. This requires `torch`, `torchvision`, and `Pillow` (PIL) to be installed.

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define the model architecture (must be the same as in the training script)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
model = SimpleCNN(num_classes).to(device)
# Make sure to use the correct path to your model file
model.load_state_dict(torch.load('Mahi/digits_model_CNN__from_png.pth', map_location=device))
model.eval()

# 3. Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 4. Create a helper function for prediction
def predict_digit(image_path):
    """
    Takes the path to an image file and returns the predicted digit.
    """
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error: Cannot open image at {image_path}")
        return None

    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and send to device

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        return class_names[predicted.item()]

# --- Example Usage ---
# predicted_digit = predict_digit('path/to/your/digit_image.png')
# if predicted_digit:
#     print(f"The predicted digit is: {predicted_digit}")

```
