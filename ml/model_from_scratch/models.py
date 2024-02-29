
"""
Author: Victor J.
Description: Function-based CNN created from scratch
Date: Winter 2023
"""

# ---------------------------------------------------------------------------- #
def model_v2(input_size=(IMG_SIZE_v2, IMG_SIZE_v2, 3), num_classes=2):
    model = tf.keras.Sequential()
    # depth, frame size are first 2 args
    # First layer of sequential model should get input_shape as arg
    # Input 302 x 302 x 3

    model.add(layers.Conv2D(
        32, 20,
        strides = 6,
        activation = activations.relu,
        input_shape = input_size,
        kernel_regularizer = tf.keras.regularizers.L2(),
        padding = "same",
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
        kernel_regularizer = tf.keras.regularizers.L2(),
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ZeroPadding2D(1))

    model.add(layers.Conv2D(
        40, 3,
        strides = 1,
        activation = activations.relu,
        kernel_regularizer = tf.keras.regularizers.L2(),
    ))
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

    optimizer = optimizers.Adam(learning_rate = 0.0001)
    loss = losses.CategoricalCrossentropy()

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ['accuracy']
    )

    model.summary()

    return model

# ---------------------------------------------------------------------------- #
def sequential_model_v1(input_size=(IMG_SIZE, IMG_SIZE, 3), num_classes=2):
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

    optimizer = optimizers.Adam(learning_rate=0.0001)
    loss = losses.CategoricalCrossentropy()

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ['accuracy'],
    )

    model.summary()

    return model
