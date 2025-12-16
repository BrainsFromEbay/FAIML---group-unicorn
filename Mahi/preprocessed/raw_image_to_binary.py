import numpy as np
import os
import cv2
import pickle

BASE_DIR = "."
IMG_SIZE = 32

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def create_data(data_type):
    """
    data_type: 'Train' or 'Validation'
    Returns: X (features/images), y (labels)
    """
    data_path = os.path.join(BASE_DIR, data_type)
    
    X = [] # Image data
    y = [] # Labels
    
    for category in CATEGORIES:
        path = os.path.join(data_path, category)
        try:
            if not os.path.exists(path):
                print(f"Warning: Folder {path} not found. Skipping.")
                continue

            class_num = CATEGORIES.index(category) 
            
            for img_name in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                    
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    
                    X.append(new_array)
                    y.append(class_num)
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Error accessing folder {category}: {e}")

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    
    print(f"Finished {data_type}. Total images: {len(X)}")
    return X, y


X_train, y_train = create_data("Train")

X_val, y_val = create_data("Validation")

data_package = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "categories": CATEGORIES
}

output_filename = "digits_data.pickle"
with open(output_filename, "wb") as f:
    pickle.dump(data_package, f)
