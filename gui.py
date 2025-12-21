import streamlit as st
import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Models


try:
    import Oyshe.prediction as oyshe_model
except ImportError as e:
    st.error(f"Failed to import Oyshe's User module: {e}")


try:
    import Jere.predict as jere_model
except ImportError as e:
    st.error(f"Failed to import Jere's User module: {e}")

try:
    import Mahi.src.inference.mlp_mnist_inference as mahi_mlp
except ImportError as e:
    st.error(f"Failed to import Mahi's MLP module: {e}")

try:
    import Mahi.src.inference.rf_mnist_inference as mahi_rf
except ImportError as e:
    st.error(f"Failed to import Mahi's RF module: {e}")

try:
    import Mahi.src.inference.cnn_raw_inference as mahi_cnn_raw
except ImportError as e:
    st.error(f"Failed to import Mahi's CNN Raw module: {e}")

try:
    import Mahi.src.inference.cnn_pickle_inference as mahi_cnn_pickle
except ImportError as e:
    st.error(f"Failed to import Mahi's CNN Pickle module: {e}")

try:
    import Mahi.src.inference.mlp_pickle_inference as mahi_mlp_pickle
except ImportError as e:
    st.error(f"Failed to import Mahi's MLP Pickle module: {e}")


# PageConfig
st.set_page_config(
    page_title="FAIML Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Title and Sidebar
st.title("🔢 FAIML Group Unicorn - Digit Recognition")
st.markdown("### Hand-written Digit Recognition System")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.radio(
        "Select Model",
        ("Mahi - CNN Raw", "Mahi - CNN Pickle", "Mahi - MLP MNIST", "Mahi - MLP Pickle", "Mahi - Random Forest", "Oyshe - HOG + Logistic Regression", "Jere - CNN")
    )
    
    st.info(f"You selected: **{model_choice}**")
    
    st.markdown("---")
    st.write("Current Working Directory:", os.getcwd())

# Helper to load models (Lazy Loading)


@st.cache_resource
def load_oyshe_model():
    return oyshe_model.load_model()

@st.cache_resource
def load_jere_model():
    # Use Jere folder as base path
    return jere_model.load_model(base_path="Jere")

@st.cache_resource
def load_mahi_mlp_model():
    return mahi_mlp.load_model()

@st.cache_resource
def load_mahi_rf_model():
    return mahi_rf.load_model()

@st.cache_resource
def load_mahi_cnn_raw_model():
    return mahi_cnn_raw.load_model()

@st.cache_resource
def load_mahi_cnn_pickle_model():
    return mahi_cnn_pickle.load_model()

@st.cache_resource
def load_mahi_mlp_pickle_model():
    return mahi_mlp_pickle.load_model()


# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    
    upload_option = st.radio("Input Method", ["Upload Image", "Select from Folder"])
    
    image_path = None
    uploaded_file = None
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    else:
        # Allow user to input folder path directly
        folder_path = st.text_input("Enter folder path", value="", placeholder="e.g., custom_test")
        
        if folder_path:
            if os.path.isdir(folder_path):
                files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if files:
                    selected_file = st.selectbox("Select an image", files)
                    if selected_file:
                        image_path = os.path.join(folder_path, selected_file)
                else:
                    st.warning("No images found in this folder.")
            else:
                st.error(f"Directory '{folder_path}' not found.")
        else:
            st.info("Please enter a folder path to browse images.")



    # Display Image
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        # Save temp for compatibility with scripts that expect paths
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = "temp_image.png"
    elif image_path:
        image = Image.open(image_path)
        st.image(image, caption=f"Selected: {os.path.basename(image_path)}", width=250)

with col2:
    st.subheader("Prediction")
    
    if image_path and st.button("Predict Digit", type="primary"):
        with st.spinner(f"Running inference with {model_choice}..."):
            try:
                prediction = None
                confidence = 0.0
                
                if "Mahi - CNN Raw" == model_choice:
                    model = load_mahi_cnn_raw_model()
                    if model:
                        prediction, confidence = mahi_cnn_raw.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's CNN Raw model.")

                elif "Mahi - CNN Pickle" == model_choice:
                    model = load_mahi_cnn_pickle_model()
                    if model:
                        prediction, confidence = mahi_cnn_pickle.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's CNN Pickle model.")
                
                elif "Mahi - MLP Pickle" == model_choice:
                    model = load_mahi_mlp_pickle_model()
                    if model:
                         prediction, confidence = mahi_mlp_pickle.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's MLP Pickle model.")

                elif "Mahi - MLP MNIST" == model_choice:
                    model = load_mahi_mlp_model()
                    if model:
                         prediction, confidence = mahi_mlp.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's MLP model.")
                
                elif "Mahi - Random Forest" in model_choice:
                    model = load_mahi_rf_model()
                    if model:
                         prediction, confidence = mahi_rf.predict_single(model, image_path)
                    else:
                        st.error("Could not load Mahi's RF model.")
                        
                elif "Oyshe" in model_choice:

                    weights, bias = load_oyshe_model()
                    prediction, confidence = oyshe_model.predict_digit(image_path, weights, bias)
                    confidence = confidence * 100 # Adjust to percentage
                    
                elif "Jere" in model_choice:
                    model = load_jere_model()
                    # Jere's predict uses GPU/device logic internally, need to reference `jere_model.device`
                    prediction, confidence = jere_model.predict_image(image_path, model, jere_model.device, show_image=False)
                
                if prediction is not None:
                    st.success(f"**Predicted Digit:** {prediction}")
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    if confidence < 50:
                        st.warning("Low confidence alert!")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                import traceback
                st.text(traceback.format_exc())
                
                # Clean up temp file
                if uploaded_file and os.path.exists("temp_image.png"):
                    os.remove("temp_image.png")

# Performance Metrics Section
st.markdown("---")
st.header("Model Performance & Statistics")


if "Mahi - CNN Raw" == model_choice:
    st.markdown("### Mahi's Model Results (CNN Raw)")
    cm_path = "Mahi/results/cnn_raw/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info("No pre-generated confusion matrix found for this model.")

elif "Mahi - CNN Pickle" == model_choice:
    st.markdown("### Mahi's Model Results (CNN Pickle)")
    cm_path = "Mahi/results/cnn_pickle/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info("No pre-generated confusion matrix found for this model.")

elif "Mahi - MLP Pickle" == model_choice:
    st.markdown("### Mahi's Model Results (MLP Pickle)")
    cm_path = "Mahi/results/mlp_pickle/confusion_matrix.png" 
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info("No pre-generated confusion matrix found for this model.")

elif "Mahi - MLP MNIST" == model_choice:
    st.markdown("### Mahi's Model Results (MLP MNIST)")
    cm_path = "Mahi/results/mlp_mnist/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info(f"Confusion Matrix not found in path ({cm_path}).")

elif "Mahi - Random Forest" in model_choice:
    st.markdown("### Mahi's Model Results (Random Forest)")
    cm_path = "Mahi/results/rf_mnist/confusion_matrix.png"
    if os.path.exists(cm_path):
         st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
         st.info(f"Confusion Matrix not found in path ({cm_path}).")

elif "Oyshe" in model_choice:
    st.markdown("### Oyshe's Model Results (HOG + Logistic Regression)")
    # Path updated to specific folder
    cm_path = "Oyshe/results/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
        st.info("Confusion Matrix not found. (Run Oyshe/prediction.py first)")

elif "Jere" in model_choice:
    st.markdown("### Jere's Model Results (CNN)")
    # Path updated to specific folder
    cm_path = "Jere/results/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=600)
    else:
        st.info(f"Confusion Matrix not found in 'Jere/results/'. ({cm_path})")

# Model Description Section
st.markdown("---")
st.header("Model Description")

description = ""
if "Mahi - CNN Raw" == model_choice:
    description = """
    **Architecture**: Simple CNN
    - **Layers**: 2 Conv Layers (16/32), MaxPool, FC(128) -> FC(10).
    - **Input**: 28x28 Raw Images (Inverted/Thresholded).
    - **Training**: Trained on raw/processed image data.
    """
elif "Mahi - CNN Pickle" == model_choice:
    description = """
    **Architecture**: Simple CNN (Deeper)
    - **Layers**: 2 Conv blocks (32 filters), 2 Conv blocks (64 filters), Dropout.
    - **Structure**: More complex feature extraction than raw model.
    - **Input**: 32x32 Resized Images.
    """
elif "Mahi - MLP MNIST" == model_choice:
    description = """
    **Architecture**: Multi-Layer Perceptron (MLP)
    - **Layers**: Input Flatten -> Dense(512) -> Dense(256) -> Output(10).
    - **Regularization**: Dropout (0.2).
    - **Training**: Standard MNIST.
    """
elif "Mahi - MLP Pickle" == model_choice:
    description = """
    **Architecture**: Multi-Layer Perceptron (MLP)
    - **Layers**: Dense(256) -> Dense(128) -> Dense(64) -> Output(10).
    - **Features**: Batch Normalization usage.
    - **Input**: 32x32 Flattened.
    """
elif "Mahi - Random Forest" in model_choice:
    description = """
    **Architecture**: Random Forest Classifier
    - **Type**: Ensemble Learning method using multiple Decision Trees.
    - **Input**: Flattened 28x28 normalized images.
    - **Advantages**: Explainable, handles non-linear data well, robust to overfitting.
    """
elif "Oyshe" in model_choice:
    description = """
    **Architecture**: HOG + Logistic Regression
    - **Feature Extraction**: Histogram of Oriented Gradients (HOG) captures edge directions/shapes.
    - **Classifier**: Logistic Regression (Linear Classifier).
    - **Pipeline**: Image -> Resize/Invert -> Compute HOG -> Classify.
    - **Purpose**: classic computer vision technique, effective for simple shapes like digits.
    """
elif "Jere" in model_choice:
    description = """
    **Architecture**: Simple CNN
    - **Structure**: 2 Conv Layers -> MaxPool -> 2 Fully Connected Layers.
    - **Optimizer**: Adam.
    - **Loss Function**: Cross Entropy Loss.
    - **Training**: Standard MNIST training pipeline (Epochs: 10).
    """

st.info(description)
