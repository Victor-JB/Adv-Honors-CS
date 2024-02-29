
"""
Author: Victor J.
Description: Function-based CNN created from scratch
Date: Winter 2023
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disables extensive tensorflow debugging msgs

import json

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.losses as losses
import tensorflow.data as data

import matplotlib.pyplot as plt
import argparse
from utils import current_milli_time, rolling_average, current_hr_time, load_dataset

IMG_SIZE = 227 # input size calculated previously; in-class arithmetic
IMG_SIZE_v2 = 302 # model v2 image sizing

BATCH_SIZE = 32 # supposed to be close to number of classes; 32 seemed better though, also power of 2

EPOCHS = 200
CHKPT_EPOCH_SAVE_FREQ = 10
LEARNING_RATE_ADJUST_RATE = 40

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", 
    "--dataset_path", 
    required = True,
    help = "Enter custom dataset path to train classifier from"
)
ap.add_argument(
    "-c", 
    "--ckpt_save_dir", 
    required = False,
    help = "Enter custom model checkpoint path",
    default = "checkpoints",
)
ap.add_argument(
    "-r", 
    "--resume_checkpoint", 
    required = False,
    help = "Enter checkpoint path from which to resume training"
)
ap.add_argument("-t", 
                "--model_type", 
                required = False,
                help = "Specify which model you want to train with (0 or 1; will change to v1 or v2)",
                default = 1,
)
ap.add_argument(
    "-p",
    "--plot-history-save-path",
    required=False,
    help="path (.json) to save training history to (will append if file exists)",
    default=f'model_histories/model_{current_hr_time()}_hist.json',
)
# ap.add_argument("-s", "--ds_size", required = False, # not currently implemented; for the future
               # help = "Resize ds by certain percent to be smaller for faster training")

args = vars(ap.parse_args())

if os.path.isdir(args['ckpt_save_dir']):
    args['ckpt_save_dir'] += current_hr_time()

# ----------------------------------------------------------------------------
def model_v2(input_size=(IMG_SIZE_v2, IMG_SIZE_v2, 3), num_classes=2, num_batches=None):
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

    if not num_batches:
        num_batches = 30
        print("\nNo batch_size provided, using a default value of 30")
    else:
        print(f"\nDecaying learning rate every {LEARNING_RATE_ADJUST_RATE} epochs ({num_batches} batches)")

    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.0004,
        decay_steps = LEARNING_RATE_ADJUST_RATE * num_batches,
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
def sequential_model_v1(input_size=(IMG_SIZE, IMG_SIZE, 3), num_classes=2, num_batches=None):
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

    if not num_batches:
        num_batches = 30
        print("\nNo batch_size provided, using a default value of 30")
    else:
        print(f"\nDecaying learning rate every {LEARNING_RATE_ADJUST_RATE} epochs ({num_batches} batches)")

    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.0004,
        decay_steps = LEARNING_RATE_ADJUST_RATE * num_batches,
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

# ---------------------------------------------------------------------------- #
def save_model_json(history, json_path):
    print(f"Saving training history to {json_path}")

    old_history = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": [],
    }
    try:
        with open(json_path, "r") as f:
            old_history = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        pass

    if old_history is not None:
        old_history["accuracy"] += history.history["accuracy"]
        old_history["loss"] += history.history["loss"]
        old_history["val_accuracy"] += history.history["val_accuracy"]
        old_history["val_loss"] += history.history["val_loss"]

    with open(json_path, "w") as f:
        json.dump(old_history, f, indent=4)

# ---------------------------------------------------------------------------- #
if int(args['model_type']) == 0:
    current_img_size = IMG_SIZE
elif int(args['model_type']) == 1:
    current_img_size = IMG_SIZE_v2
else:
    raise ValueError('"--model_type" argument provided is not int 0 or 1')

ds_train, ds_test, num_classes = load_dataset(
                        args['dataset_path'], 
                        current_img_size,
                        BATCH_SIZE,
)

if args['resume_checkpoint']:
    model = tf.keras.models.load_model(args['resume_checkpoint'])
    print(f"\nModel at '{args['resume_checkpoint']}' loaded successfully")
elif int(args['model_type']) == 0:
    model = sequential_model_v1()
    print("\nv1 Model created and loaded successfully")
else:
    model = model_v2(num_batches=len(ds_train))
    print("\nv2 Model created and loaded successfully")

print(f"\nSaving checkpoints at '{args['ckpt_save_dir']}'; saving every {CHKPT_EPOCH_SAVE_FREQ} epochs\n")

callback = [
    callbacks.ModelCheckpoint(
        filepath = args['ckpt_save_dir'] + '/checkpoint_{epoch:02d}',
        # save_best_only = True,
        verbose = 1,
        save_freq = CHKPT_EPOCH_SAVE_FREQ * len(ds_train),
    )
]

print(f"Starting model training for {EPOCHS} epochs...\n") # console formatting ;)
history = model.fit(
    ds_train,
    epochs = EPOCHS,
    # batch_size = BATCH_SIZE, <-- batching done in preprocessing
    verbose = 1,
    validation_data = ds_test,
    callbacks = callback,
)

save_path = f"{args['ckpt_save_dir']}/recent_model_ep{EPOCHS}"

print(f"Saving model to {save_path}")
model.save(save_path)

save_model_json(history, args['plot_history_save_path'])
