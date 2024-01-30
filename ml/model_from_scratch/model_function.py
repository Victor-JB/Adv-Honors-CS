
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
from utils import current_milli_time, rolling_average

IMG_SIZE = 227
BATCH_SIZE = 32 # supposed to be close to number of classes; 32 seemed better though, also power of 2
EPOCHS = 1
DEF_DATASET_NAME = "datasets/krunker"
CHKPT_EPOCH_SAVE_FREQ = 10
CHECKPOINT_DIR = 'checkpoints'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path", required = False,
               help = "Enter custom dataset path to train classifier from")
ap.add_argument("-c", "--ckpt_save_dir", required = False,
               help = "Enter custom model checkpoint path")
ap.add_argument("-r", "--resume_checkpoint", required = False,
               help = "Enter checkpoint path from which to resume training")
# ap.add_argument("-s", "--ds_size", required = False, # not currently implemented; for the future
               # help = "Resize ds by certain percent to be smaller for faster training")
args = vars(ap.parse_args())

def plot_hist(history):
    # 5 chosen below for r_avg arbitrarily; is window which rolling avg computes
    """
    plt.plot(EPOCHS, rolling_average(history.history["accuracy"], 5),
             'bo', label='Training acc'
            )
    plt.plot(EPOCHS, rolling_average(history.history["val_accuracy"], 5),
             'b', label='Validation acc'
             )
    """
    plt.plot(EPOCHS, history.history["accuracy"],
             'bo', label='Training acc'
            )
    plt.plot(EPOCHS, history.history["val_accuracy"],
             'b', label='Validation acc'
             )

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('1000_acc_epochs.png')

    plt.figure()

    """
    plt.plot(EPOCHS, rolling_average(history.history["loss"], 5),
             'bo', label='Training loss'
             )
    plt.plot(EPOCHS, rolling_average(history.history["val_loss"], 5),
             'b', label='Validation loss'
             )
    """
    plt.plot(EPOCHS, history.history["loss"],
             'bo', label='Training loss'
             )
    plt.plot(EPOCHS, history.history["val_loss"],
             'b', label='Validation loss'
             )

    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('1000_loss_epochs.png')
    plt.show()

def load_dataset(DS_PATH):
    # will handle ds path validation for me
    ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
        DS_PATH,
        label_mode = 'categorical',
        image_size = (IMG_SIZE, IMG_SIZE),
        seed = 18181,
        validation_split = 0.30,
        batch_size = BATCH_SIZE,
        subset = 'both',
    )

    print(f"\nDataset images have been resized to ({IMG_SIZE}, {IMG_SIZE})")

    NUM_CLASSES = len(ds_train.class_names)

    print(f"\nDataset has been loaded; contains {NUM_CLASSES} classes")

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    print("\nDataset has been resized to uniform IMG_SIZE, labels have been put \
into one-hot (categorical) encoding, the dataset has been batched.")

    return ds_train, ds_test

def sequential_model(input_size):
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

if args['dataset_path']:
    ds_train, ds_test = load_dataset(args['dataset_path'])
else:
    ds_train, ds_test = load_dataset(DEF_DATASET_NAME)

if args['resume_checkpoint']:
    model = tf.keras.models.load_model(args['resume_checkpoint'])
    print(f"\nModel at '{args['resume_checkpoint']}' loaded successfully")
else:
    input_size = (IMG_SIZE, IMG_SIZE, 3)
    model = sequential_model(input_size)

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

print(f"\nStarting model training for {EPOCHS} epochs...\n") # console formatting ;)
history = model.fit(
    ds_train,
    epochs = EPOCHS,
    # batch_size = BATCH_SIZE, <-- batching done in preprocessing
    verbose = 1,
    validation_data = ds_test,
    callbacks = callback,
)

plot_hist(history)
