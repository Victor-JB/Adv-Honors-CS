
"""
Author: Victor J.
Description: Function-based CNN created from scratch
Date: Winter 2023
"""

import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# import data here

def sequential_model(input_size):
    model = tf.keras.Sequential()
    # depth, frame size, stride
    # first layer of sequential model needs to include input_shape arg
    # Input: 227 x 227 x  <-- take all of these with grain of salt b/c args could all change
    model.add(layers.Conv2D(
        12,
        11,
        strides=4,
        activation=activations.relu,
        input_shape=input_size,
    ))
    # Size after conv: 55 x 55 x 12

    # only needed if no activation in first cov2d
    # model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D(
        pool_size=3,
        strides=2,
    ))
    # Size after pool: 27 x 27 x 12

    model.add(layers.Conv2D(
        18,
        3,
        strides=1,
        activation=activations.relu,
    ))
    # size after conv: 25 x 25 x 18

    model.add(layers.MaxPooling2D(
        pool_size=3,
        strides=2,
    ))
    # Size after pool: 12 x 12 x 18

    model.add(layers.Flatten())
    # size: 2592

    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(128, activation=activations.relu))
    model.add(layers.Dense(32, activation=activations.relu))

    # size of last Dense layer must match # of classes
    model.add(layers.Dense(5, activation=activations.softmax))

    optimizer = optimizers.Adam(learning_rate=0.0001)
    loss = losses.CategoricalCrossentropy()

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ['accuracy'],
    )

    model.summary()

    return model


# ---------------------------------------------------------------------------- #
# input size calculated previously; in-class arithmetic

input_size = (227, 227, 3)
model = sequential_model(input_size)

print("\nModel created:", model)
