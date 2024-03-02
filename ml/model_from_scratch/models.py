
"""
Author: Victor J.
Description: Function-based CNN created from scratch
Date: Winter 2023
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# ----------------------------------------------------------------------------
def seq_modelv2(input_shape, num_classes=2, init_lr=0.00005, lr_adjust=None):
    model = tf.keras.Sequential()
    # depth, frame size are first 2 args
    # First layer of sequential model should get input_shape as arg
    # Input given by input_shape, something like 500x1000

    model.add(layers.Conv2D(
        32, 20,
        strides = 6,
        activation = activations.relu,
        input_shape = input_shape,
        kernel_regularizer = tf.keras.regularizers.L2(),
        ))
    model.add(layers.BatchNormalization())
    # 48 x 48 x 32

    model.add(layers.Conv2D(
        32, 3,
        strides = 1,
        activation = activations.relu,
        kernel_regularizer = tf.keras.regularizers.L2(),
        ))
    model.add(layers.BatchNormalization())
    model.add(layers.ZeroPadding2D(1))

    model.add(layers.Conv2D(
        40, 3,
        strides = 1,
        activation = activations.relu,
        kernel_regularizer = tf.keras.regularizers.L2(),))
    model.add(layers.BatchNormalization())
    model.add(layers.ZeroPadding2D(1))

    model.add(layers.Conv2D(
        40, 3,
        strides = 1,
        activation = activations.relu,
        kernel_regularizer = tf.keras.regularizers.L2(),))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(
        pool_size = 3,
        strides = 2,
    ))

    model.add(layers.ZeroPadding2D(1))

    model.add(layers.Conv2D(
        40, 3,
        strides = 2,
        activation = activations.relu,
    ))

    model.add(layers.MaxPooling2D(
        pool_size = 3,
        strides = 2,
    ))

    model.add(layers.Flatten())

    # Size 6760?? need to recalculate... i don't think that's true...
    model.add(layers.Dense(1024, activation = activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation = activations.relu,))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation = activations.relu,))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation = activations.relu,))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(64, activation = activations.relu,))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(32, activation = activations.relu,))
    model.add(layers.Dropout(0.5))

    # size of last Dense layer must match # of classes
    model.add(layers.Dense(num_classes, activation = activations.softmax,))

    if not lr_adjust:
        lr_adjust = 300
        print("\nNo lr_adjust batch num provided, using a default value of 300")
    else:
        print(f"\nDecaying learning rate every {lr_adjust} batches)")

    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = init_lr,
        decay_steps = lr_adjust, # number of batches before decaying learning rate
        decay_rate = 0.1,
    )
    optimizer = optimizers.Adam(learning_rate=lr_scheduler)
    loss = losses.CategoricalCrossentropy()

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ['accuracy']
    )

    model.summary()

    return model

# ---------------------------------------------------------------------------- #
def seq_modelv1(input_size=(300, 300, 3), num_classes=2, init_lr=0.0005, lr_adjust=None):
    model = tf.keras.Sequential()
    # depth, frame size, stride
    # first layer of sequential model needs to include input_shape arg
    # Input: 227 x 227 x  <-- take all of these with grain of salt b/c args could all change
    model.add(layers.Conv2D(
        35,
        11,
        strides=4,
        activation=activations.relu,
        input_shape=input_size,
    ))
    # Size after conv: 55 x 55 x 35

    # only needed if no activation in first cov2d
    # model.add(layers.Activation(activations.relu))

    model.add(layers.BatchNormalization())

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
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation=activations.relu))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dropout(0.5))

    # size of last Dense layer must match # of classes
    model.add(layers.Dense(num_classes, activation=activations.softmax))

    if not lr_adjust:
        lr_adjust = 300
        print("\nNo lr_adjust batch num provided, using a default value of 300")
    else:
        print(f"\nDecaying learning rate every {lr_adjust} batches)")

    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = init_lr,
        decay_steps = lr_adjust, # number of batches before decaying learning rate
        decay_rate = 0.1,
    )
    optimizer = optimizers.Adam(learning_rate=lr_scheduler)
    loss = losses.CategoricalCrossentropy()

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ['accuracy'],
    )

    model.summary()

    return model
