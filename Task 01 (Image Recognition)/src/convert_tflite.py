import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"C:\CodeAlpha_Image-Recognition\models\mnist_model.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("mnist_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite Model Saved!")
