import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import models 
from tensorflow.keras import layers

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Increased filters
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Increased units
        layers.Dense(3)  # Output layer with x, y, radius dimensions
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
