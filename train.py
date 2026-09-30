import tensorflow as tf
import numpy as np
import os

print("Loading MNIST dataset...")

# Load the MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Create the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    # 10 output neurons = digits 0 to 9
    tf.keras.layers.Dense(10, activation="softmax")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

print("\nTraining model...")

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

print("\nEvaluating model...")

# Evaluate using test data
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)

# Create model folder in the same folder as train.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Save trained model
model.save(os.path.join(MODEL_DIR, "digit_model.keras"))

print("\nModel saved successfully!")
print("Location:", os.path.join(MODEL_DIR, "digit_model.keras"))