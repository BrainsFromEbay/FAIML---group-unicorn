import cv2
import numpy as np
import os

def preprocess_image_for_mlp(img_path):
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    img_np = cv2.resize(img_np, (32, 32))

    img_np = 255 - img_np

    img_np[img_np < 50] = 0
    
    return img_np

image_path = 'custom_test/seven.png'

preprocessed_img = preprocess_image_for_mlp(image_path)

np.set_printoptions(linewidth=200, edgeitems=4)
with np.printoptions(threshold=np.inf):
    print(preprocessed_img)
