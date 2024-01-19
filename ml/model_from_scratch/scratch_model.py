
"""
Author: Victor J.
Description: CNN created from scratch
Date: Winter 2023
"""

import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# skipping data acquisition now

# VERSION 1: Class-based model
class Model():
    def __init__(self, input_size):
        self.model = tf.keras.Sequential()
        # depth, frame size, stride
        # first layer of sequential model needs to include input_shape arg
        # Input: 227 x 227 x  <-- take all of these with grain of salt b/c args could all change
        self.model.add(layers.Conv2D(
            12,
            11,
            strides=4,
            activation=activations.relu,
            input_shape=input_size,
        ))
        # Size after conv: 55 x 55 x 12

        # only needed if no activation in first cov2d
        # self.model.add(layers.Activation(activations.relu))

        self.model.add(layers.MaxPooling2D(
            pool_size=3,
            strides=2,
        ))
        # Size after pool: 27 x 27 x 12

        self.model.add(layers.Conv2D(
            18,
            3,
            strides=1,
            activation=activations.relu,
            input_shape=input_size,
        ))
        # size after conv: 25 x 25 x 18

        self.model.add(layers.MaxPooling2D(
            pool_size=3,
            strides=2,
        ))
        # Size after pool: 12 x 12 x 18

        self.model.add(layers.Flatten())
        # size: 2592

        self.model.add(layers.Dense(512, activation=activations.relu))
        self.model.add(layers.Dense(128, activation=activations.relu))
        self.model.add(layers.Dense(32, activation=activations.relu))

        # size of last Dense layer must match # of classes
        self.model.add(layers.Dense(5, activation=activations.softmax))

        self.optimizer = optimizers.Adam(learning_rate=0.0001)
        self.loss = losses.CategoricalCrossentropy()

        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )


# ---------------------------------------------------------------------------- #
# input size calculated previously; in-class arithmetic
model = Model((227, 277, 3))
