
"""
Author: Victor J.
Description: Function-based CNN created from scratch
Date: Winter 2023
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disables extensive tensorflow debugging msgs

import json

from models import seq_modelv1, seq_modelv2

import tensorflow.keras.callbacks as callbacks

import matplotlib.pyplot as plt
import argparse
from utils import current_milli_time, rolling_average, current_hr_time, load_dataset

# height x width, per tensorflow
IMG_SHAPE = (500, 500, 3) # arbitrarily chosen; keeps enough data whilst not making it too large

BATCH_SIZE = 64 # supposed to be close to number of classes; 32 seemed better though, also power of 2

EPOCHS = 200
CHKPT_EPOCH_SAVE_FREQ = 10

INITIAL_LEARNING_RATE = 0.00005
LEARNINGR_EPCH_ADJUST_RATE = 40

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
                help = "Specify which model you want to train with (1 or 2; will change to v1 or v2)",
                default = 2,
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

ds_train, ds_test, num_classes = load_dataset(
                        args['dataset_path'],
                        IMG_SHAPE[:-1], # taking out third dimension from tuple
                        BATCH_SIZE,
)

LEARNING_RATE_NUM_BATCHES = LEARNINGR_EPCH_ADJUST_RATE * len(ds_train)

if args['resume_checkpoint']:
    model = tf.keras.models.load_model(args['resume_checkpoint'])
    print(f"\nModel at '{args['resume_checkpoint']}' loaded successfully")
elif int(args['model_type']) == 1:
    model = seq_modelv1(
        input_shape = IMG_SHAPE,
        num_classes = num_classes,
        init_lr = INITIAL_LEARNING_RATE,
        lr_adjust = LEARNING_RATE_NUM_BATCHES,
    )
    print("\nv1 Model created and loaded successfully")
elif int(args['model_type']) == 2:
    model = seq_modelv2(
        input_shape = IMG_SHAPE,
        num_classes = num_classes,
        init_lr = INITIAL_LEARNING_RATE,
        lr_adjust = LEARNING_RATE_NUM_BATCHES,
    )
    print("\nv2 Model created and loaded successfully")
else:
    raise ValueError('"--model_type" argument provided is not int 1 or 2')

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
