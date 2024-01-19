
"""
Author: Victor J.
Description: Non-sequential function-based CNN created from scratch
Date: Winter 2023
"""

import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# import data here

inputs = tf.keras.Input(shape = (227, 227, 3))

# first layer
outputs = layers.Conv2D(
    12,
    11,
    strides=4,
    activation=activations.relu,
))(inputs)
# Size after conv: 55 x 55 x 12

outputs = layers.MaxPooling2D(pool_size=3, strides=2)(outputs)
...

model = tf.keras.Model(inputs, outputs)

loss = losses.CategoricalCrossentropy()
optimizer = optmizers.Adam(learning_rate = 0.0001)

model.compile(
    loss = loss,
    optimizer = optmizer,
    metrics = ['accuracy'],
)

model.fit(...)
