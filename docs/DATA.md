# Datasets

This project utilizes two main sources of data for training and testing the models: the well-known MNIST dataset and a custom set of test images.

## The MNIST Dataset

The MNIST dataset is a large database of handwritten digits that is commonly used for training and testing in the field of machine learning. It contains 60,000 training images and 10,000 testing images.

*   **Source:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
*   **Format:** The images are grayscale and have a size of 28x28 pixels.
*   **Usage:** The MNIST dataset is the primary dataset used for training and evaluating the models in this project. The training scripts in the `Jere/`, `Mahi/`, and `Oyshe/` directories all use this dataset.

The dataset is expected to be located in the `data/MNIST/` directory. Some of the training scripts may download it automatically if it's not found.

## Custom Test Images

To further evaluate the models' performance on real-world, "in-the-wild" data, a custom set of test images is provided in the `custom_test/` directory.

*   **Source:** These images have been created manually.
*   **Format:** The images are PNG files, and the filename indicates the digit they represent (e.g., `7.png`, `3(1).png`).
*   **Usage:** These images are used by the inference scripts (`predict.py`, `*_inference.py`) and the main `gui.py` to test the models with images that were not part of the MNIST dataset. This helps to assess the generalization capabilities of the models.

The images in the `custom_test` directory have varying backgrounds and drawing styles, making them a good test of the models' robustness.
