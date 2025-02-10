import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)  # Reshape for CNN

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\CodeAlpha_Image-Recognition\models\mnist_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference on first 10 test images
correct = 0
for i in range(10):
    img = np.expand_dims(x_test[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))
    print(f"Image {i}: Predicted {prediction}, Actual {y_test[i]}")
    
    if prediction == y_test[i]:
        correct += 1

accuracy = (correct / 10) * 100
print(f"\nTest Accuracy on 10 samples: {accuracy:.2f}%")
