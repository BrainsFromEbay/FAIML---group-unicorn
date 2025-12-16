import pickle
import numpy as np
import os

# Paths
original_pickle_path = "Mahi/digits_data.pickle"
cleaned_pickle_path = "Mahi/digits_data_cleaned.pickle"

print("Loading the original pickle file...")
with open(original_pickle_path, 'rb') as f:
    data = pickle.load(f)

# Expected keys based on your earlier prints
# These should be present: X_train, y_train, X_val, y_val, categories
print("Keys in pickle:", list(data.keys()))
print(f"X_train original dtype: {data['X_train'].dtype}")
print(f"X_train original min/max: {data['X_train'].min()} / {data['X_train'].max()}")

# Threshold: set all values < 50 to 0
# We operate directly on the arrays (uint8) - this is memory efficient
threshold = 50

print(f"Applying threshold (< {threshold} → 0) to X_train and X_val...")

# Create masks and apply in-place to save memory
mask_train = data['X_train'] < threshold
mask_val = data['X_val'] < threshold

data['X_train'][mask_train] = 0
data['X_val'][mask_val] = 0

# Optional: confirm the change
print(f"After cleaning:")
print(f"X_train min/max: {data['X_train'].min()} / {data['X_train'].max()}")
print(f"X_val   min/max: {data['X_val'].min()} / {data['X_val'].max()}")

# Count how many pixels were zeroed (just for info)
pixels_zeroed_train = mask_train.sum()
pixels_zeroed_val = mask_val.sum()
total_pixels_train = data['X_train'].size
total_pixels_val = data['X_val'].size

print(f"Pixels set to 0 in X_train: {pixels_zeroed_train:,} / {total_pixels_train:,} "
      f"({100 * pixels_zeroed_train / total_pixels_train:.2f}%)")
print(f"Pixels set to 0 in X_val:   {pixels_zeroed_val:,} / {total_pixels_val:,} "
      f"({100 * pixels_zeroed_val / total_pixels_val:.2f}%)")

# Save the cleaned version
print(f"\nSaving cleaned dataset to '{cleaned_pickle_path}'...")
with open(cleaned_pickle_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! Cleaned dataset saved.")
print("\nYou can now use 'digits_data_cleaned.pickle' for training with PyTorch.")