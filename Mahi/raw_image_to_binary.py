import numpy as np
import os
import cv2
import pickle

# --- Configuration ---
# Set the path where your folders are located
BASE_DIR = "."  # Change this if your script is not in the same folder as 'Train'/'Validation'
IMG_SIZE = 32   # Resize images to 32x32 (standardizing size is required for models)

# We only want digits for now
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def create_data(data_type):
    """
    data_type: 'Train' or 'Validation'
    Returns: X (features/images), y (labels)
    """
    data_path = os.path.join(BASE_DIR, data_type)
    
    X = [] # Image data
    y = [] # Labels
    
    print(f"Processing {data_type} data...")

    for category in CATEGORIES:
        path = os.path.join(data_path, category)
        try:
            # Check if the folder exists to avoid errors
            if not os.path.exists(path):
                print(f"Warning: Folder {path} not found. Skipping.")
                continue

            # Get the class index (e.g., 0 for "0", 1 for "1")
            class_num = CATEGORIES.index(category) 
            
            for img_name in os.listdir(path):
                try:
                    # 1. Read the image in Grayscale
                    img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                    
                    # 2. Resize the image (Ensure all inputs are the same size)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    
                    # 3. Append to our list
                    X.append(new_array)
                    y.append(class_num)
                except Exception as e:
                    # Should an image be corrupt, we skip it
                    pass
        except Exception as e:
            print(f"Error accessing folder {category}: {e}")

    # Convert to numpy arrays
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Reshape for CNN input (batch, height, width, channels)
    y = np.array(y)
    
    print(f"Finished {data_type}. Total images: {len(X)}")
    return X, y

# --- Execution ---

# 1. Process Training Data
X_train, y_train = create_data("Train")

# 2. Process Validation Data
X_val, y_val = create_data("Validation")

# --- Saving to Pickle ---

print("Saving data to pickle files...")

# We create a dictionary to hold everything nicely
data_package = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "categories": CATEGORIES # Save class names so we don't forget them
}

# Save as a single binary file
output_filename = "digits_data.pickle"
with open(output_filename, "wb") as f:
    pickle.dump(data_package, f)

print(f"Success! Data saved to {output_filename}")