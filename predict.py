import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model("mnist_model.h5")
print("Model loaded successfully.")

# Load MNIST dataset (only test set)
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize and reshape test images
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Select a test image (e.g., first image)
index = 0  # Change this to test different images
test_image = x_test[index]
true_label = y_test[index]

# Predict the class
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_class = np.argmax(prediction)

# Display the image with prediction
plt.imshow(test_image.squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted_class}, Actual: {true_label}")
plt.axis("off")
plt.show()

print(f"Predicted class: {predicted_class}, Actual class: {true_label}")
