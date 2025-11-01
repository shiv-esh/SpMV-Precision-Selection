import streamlit as st
import tensorflow as tf
from scipy.io import mmread
import numpy as np
import os
from PIL import Image

model = tf.keras.models.load_model('hist_model.h5')

def histnorm(A, r, BINS):
    """
    Creates a row-based histogram representation of the input matrix A.

    Args:
        A: The input matrix.
        r: The number of rows in the output histogram.
        BINS: The number of bins in each row of the histogram.

    Returns:
        The histogram matrix R.
    """
    R = np.zeros((r, BINS), dtype=int)
    ScaleRatio = A.shape[0] / r
    MaxDim = max(A.shape[0], A.shape[1])

    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            if A[row, col] != 0:
                row_index = int(row / ScaleRatio)
                bin_index = int(BINS * (row / MaxDim - col / MaxDim))
                R[row_index, bin_index] += 1

    return R

def preprocess_mtx(file, r=128, BINS=128, image_size=(128, 128)):
    """
    Preprocesses a .mtx file and converts it into a histogram representation.

    Args:
        file: Path to the .mtx file.
        r: Number of rows in the histogram.
        BINS: Number of bins in each row of the histogram.
        image_size: Target size for resizing the histogram matrix.

    Returns:
        Preprocessed tensor suitable for model input.
    """
    # Load the .mtx file as a sparse matrix
    sparse_matrix = mmread(file)
    
    # Convert the sparse matrix to a dense matrix
    dense_matrix = sparse_matrix.toarray()
    
    # Generate histogram representation
    histogram_matrix = histnorm(dense_matrix, r, BINS)
    
    # Normalize the histogram to [0, 255]
    max_value = histogram_matrix.max()
    if max_value > 0:  # Avoid division by zero
        histogram_matrix = (histogram_matrix / max_value) * 255
    histogram_matrix = histogram_matrix.astype(np.uint8)
    
    # Ensure the histogram_matrix has 3 dimensions (H, W, C)
    if len(histogram_matrix.shape) == 2:  # Add channel dimension for grayscale
        histogram_matrix = np.expand_dims(histogram_matrix, axis=-1)
    
    # Convert histogram to a tensor
    histogram_tensor = tf.convert_to_tensor(histogram_matrix, dtype=tf.float32)
    
    # Resize to match the model's input size
    resized_tensor = tf.image.resize(histogram_tensor, image_size)
    
    # Add batch dimension to make it 4D (B, H, W, C)
    input_tensor = tf.expand_dims(resized_tensor, axis=0)
    
    return input_tensor


st.title("MTX File Classification")
st.write("Upload a `.mtx` file to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose a .mtx file", type=["mtx"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp.mtx")
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    try:
        # Preprocess the .mtx file
        input_data = preprocess_mtx(temp_file_path)

        # Predict the class
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions)

        # Display the result
        st.success(f"The predicted class is: {predicted_class}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)