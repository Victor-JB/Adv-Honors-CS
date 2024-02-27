
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

EPOCHS = 80
CHKPT_EPOCH_SAVE_FREQ = 5
CHECKPOINT_DIR = 'checkpoints'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path", required = True,
               help = "Enter custom dataset path to train classifier from")
ap.add_argument("-c", "--ckpt_save_dir", required = False,
               help = "Enter custom model checkpoint path")
ap.add_argument("-r", "--resume_checkpoint", required = False,
               help = "Enter checkpoint path from which to resume training")
ap.add_argument("-t",
                "--model_type",
                required = False,
                help = "Specify which model you want to train with (0 or 1; will change to v1 or v2)",
                default = 0,
)
ap.add_argument(
    "-p",
    "--plot-history-save-path",
    required=False,
    help="path (.json) to save training history to (will append if file exists)",
    default=f'model_{current_hr_time()}_hist.json',
)
# ap.add_argument("-s", "--ds_size", required = False, # not currently implemented; for the future
               # help = "Resize ds by certain percent to be smaller for faster training")

args = vars(ap.parse_args())

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
    model = model_v2()
    print("\nv2 Model created and loaded successfully")

if args['ckpt_save_dir']:
    CHECKPOINT_DIR = args['ckpt_save_dir']

elif os.path.isdir(CHECKPOINT_DIR): # to avoid writing over checkpoints that already exist
    CHECKPOINT_DIR = 'checkpoints_' + str(current_milli_time())[10:]

print(f"\nSaving checkpoints at '{CHECKPOINT_DIR}'; saving every {CHKPT_EPOCH_SAVE_FREQ} epochs\n")

callback = [
    callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_DIR + '/checkpoint_{epoch:02d}',
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

save_path = f"{CHECKPOINT_DIR}/recent_model_ep{EPOCHS}"

print(f"Saving model to {save_path}")
model.save(save_path)

save_model_json(history, args['plot_history_save_path'])
