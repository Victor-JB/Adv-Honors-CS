
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


def current_milli_time():
    return round(time.time() * 1000)

def rolling_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# ---------------------------------------------------------------------------- #
def load_dataset(DS_PATH, IMG_SIZE, BATCH_SIZE, val_split=.6):
    # will handle ds path validation for me
    print(DS_PATH)
    ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
        DS_PATH,
        label_mode = 'categorical',
        image_size = (IMG_SIZE, IMG_SIZE),
        seed = 18181,
        validation_split = val_split,
        batch_size = BATCH_SIZE,
        subset = 'both',
    )

    print(f"\nDataset images have been resized to ({IMG_SIZE}, {IMG_SIZE})")

    NUM_CLASSES = len(ds_train.class_names)

    print(f"\nDataset has been loaded; contains {NUM_CLASSES} classes")

    ds_train = ds_train.cache().prefetch(buffer_size = data.AUTOTUNE)
    ds_test = ds_test.cache().prefetch(buffer_size = data.AUTOTUNE)

    print("\nDataset has been resized to uniform IMG_SIZE, labels have been put \
into one-hot (categorical) encoding, the dataset has been batched.")

    return ds_train, ds_test, NUM_CLASSES