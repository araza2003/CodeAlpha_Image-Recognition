import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../models/best_mnist_model.h5")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Fix the warning

def predict_custom_image(image_path):
    """Preprocess and predict a custom MNIST image."""
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found!")
        return
    
    # Resize to 28x28 (Ensure same size as MNIST)
    img = cv2.resize(img, (28, 28))

    # Invert colors if the background is black (MNIST digits are white on black)
    img = cv2.bitwise_not(img)

    # Normalize pixel values (0-255 → 0-1)
    img = img / 255.0

    # Reshape for model input (1 sample, 28x28, 1 channel)
    img = img.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    print(f"Predicted: {predicted_digit}")

# Test with a custom image
predict_custom_image(r"C:\CodeAlpha_Image-Recognition\dataset\9.png")
