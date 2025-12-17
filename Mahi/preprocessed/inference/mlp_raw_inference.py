import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2

# Use the same device as in training, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Define the exact same model architecture as in the training script
class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# 2. Define the exact same image transform as in the training script
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 3. Load the trained model
model_path = 'Mahi/raw_image/MLP_from_raw_image.pth'
model = SimpleMLP()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 4. Define a helper function for prediction
def predict_digit(image_path, model, transform):
    """Loads an image, preprocesses it, and predicts the digit."""
    # Read image in grayscale
    image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert colors: in many datasets, digits are white on a black background.
    # The training data for this model likely followed this convention.
    # The custom_test images are black on white, so they need to be inverted.
    image_np = 255 - image_np

    # Add a thresholding step to binarize the image, removing noise.
    # This makes the image more similar to the likely training data.
    image_np[image_np < 50] = 0

    # Convert to PIL Image to use torchvision transforms
    image_pil = Image.fromarray(image_np)

    # Apply the transformations and add a batch dimension
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# 5. Loop through images in the custom_test folder and predict
folder = 'custom_test'
files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

print(f"Found {len(files)} images in '{folder}'.\n")
print(f"Using model: {model_path}")
print("-" * 40)


for f in sorted(files):
    img_path = os.path.join(folder, f)
    prediction = predict_digit(img_path, model, transform)

    # Get expected digit from filename
    name = f.split('.')[0].strip()
    try:
        expected = int(name)
    except ValueError:
        expected = "Unknown"

    print(f"File: {f}")
    print(f"Expected digit: {expected}")
    print(f"Predicted digit: {prediction}")
    print(f"Match: {'Yes' if expected == prediction else 'No'}")
    print("-" * 40)

print("\nInference complete.")
