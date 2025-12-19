import pickle
import numpy as np
import os

original_pickle_path = "Mahi/src/data_preprocessing/digits_data.pickle"
cleaned_pickle_path = "Mahi/src/data_preprocessing/digits_data_cleaned.pickle"

with open(original_pickle_path, 'rb') as f:
    data = pickle.load(f)

threshold = 50

mask_train = data['X_train'] < threshold
mask_val = data['X_val'] < threshold

data['X_train'][mask_train] = 0
data['X_val'][mask_val] = 0

pixels_zeroed_train = mask_train.sum()
pixels_zeroed_val = mask_val.sum()
total_pixels_train = data['X_train'].size
total_pixels_val = data['X_val'].size

with open(cleaned_pickle_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
