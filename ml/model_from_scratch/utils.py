
"""
Author: Victor J.
Description: Misc useful functions used across multiple files, all aggregated
here for cleanliness and reusability's sake
Date: Winter 2023
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow.data as data
from datetime import datetime

# ---------------------------------------------------------------------------- #
def current_milli_time():
    return round(time.time() * 1000)

# ---------------------------------------------------------------------------- #
def current_hr_time():
    return datetime.now().strftime("%H-%M-%S")

# ---------------------------------------------------------------------------- #
def rolling_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# ---------------------------------------------------------------------------- #
def load_dataset(DS_PATH, IMG_SHAPE, BATCH_SIZE, val_split=0.3):
    # will handle ds path validation for me
    print("Loading images from dataset path," DS_PATH, "...")
    ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
        DS_PATH,
        label_mode = 'categorical',
        image_size = IMG_SHAPE,
        seed = 18181,
        validation_split = val_split,
        batch_size = BATCH_SIZE,
        subset = 'both',
    )

    print(f"\nDataset images have been resized to {IMG_SHAPE}")

    NUM_CLASSES = len(ds_train.class_names)

    print(f"\nDataset has been loaded; contains {NUM_CLASSES} classes")

    ds_train = ds_train.cache().prefetch(buffer_size = data.AUTOTUNE)
    ds_test = ds_test.cache().prefetch(buffer_size = data.AUTOTUNE)

    print("\nDataset has been resized to uniform IMG_SIZE, labels have been put \
into one-hot (categorical) encoding, the dataset has been batched.")

    return ds_train, ds_test, NUM_CLASSES