import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"C:\CodeAlpha_Image-Recognition\models\mnist_model.h5")  # Update the path if needed

# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize the test images (0-255 → 0-1)
x_test = x_test / 255.0

# Reshape for CNN input (28x28 → 28x28x1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Evaluate the model on the full test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Get predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified images
misclassified_indices = np.where(predicted_labels != y_test)[0]

# Display first 10 misclassified images
plt.figure(figsize=(10, 5))
for i, idx in enumerate(misclassified_indices[:10]):  # Show first 10 misclassified images
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predicted_labels[idx]}, Actual: {y_test[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

print(f"\nTotal Incorrect Predictions: {len(misclassified_indices)}")

# Analyze misclassified digits (focus on 0, 4, 6, 8, 9)
target_misclassified = [0, 4, 6, 8, 9]
filtered_misclassified = [idx for idx in misclassified_indices if y_test[idx] in target_misclassified]

print(f"Total Incorrect Predictions for {target_misclassified}: {len(filtered_misclassified)}")
