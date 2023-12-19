
"""
Author: Victor J.
Description: Transfer learning from TensorFlow EfficientNet image classification model,
first go :)
Date: Winter 2023
"""

import os
import numpy as np
import tensorflow_datasets as tfds # for testing with Stanford Dog dataset
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.losses as losses
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.applications import EfficientNetB3
import math
from utils import current_milli_time, rolling_average
import argparse

# doesn't seem to be working...
tf.get_logger().setLevel('ERROR') # disable those pesky tf warnings

# IMG_SIZE is determined by EfficientNet model choice; B3, in this case
IMG_SIZE = 300
BATCH_SIZE = 32 # supposed to be close to number of classes; 32 seemed better though, also power of 2
EPOCHS = 1000
DEFAULT_DATASET_NAME = "stanford_dogs"
CHKPT_EPOCH_SAVE_FREQ = 10

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path", required = False,
               help = "Enter custom dataset path to train classifier from")
ap.add_argument("-c", "--CHECKPOINT_DIR", required = False,
               help = "Enter custom model checkpoint path")
ap.add_argument("-r", "--resume_checkpoint", required = False,
               help = "Enter checkpoint path fron which to resume training")
args = vars(ap.parse_args())

# applying certain transformations to my images to augment them and increase ds size
IMG_AUGMENTATION_LAYERS = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]

def format_label(label_info, label):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]

def display_data_sample(ds_train, label_info):
    for i, (image, label) in enumerate(ds_train.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title("{}".format(format_label(label_info, label)))
        plt.axis("off")

    plt.show()

def img_augmentation(images):
    for layer in IMG_AUGMENTATION_LAYERS:
        images = layer(images)

    return images

def display_aug_data_sample(ds_train, label_info):
    for image, label in ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
            aug_img = np.array(aug_img)
            plt.imshow(aug_img[0].astype("uint8"))
            plt.title("{}".format(format_label(label_info, label)))
            plt.axis("off")

    plt.show()

# One-hot / categorical encoding only for tfds data
def input_preprocess_tfds_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# only augmenting since image_ds_from_dir already does one-hot encoding
def input_preprocess_custom_train(image, label):
    image = img_augmentation(image)
    return image, label

def input_preprocess_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

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

def load_dataset(DS_PATH=None):

    global NUM_CLASSES # need to do this for input_preprocess functions--not sure of better way
    if DS_PATH:
        # will handle ds path validation for me
        ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
            DS_PATH,
            label_mode = 'categorical',
            image_size = (IMG_SIZE, IMG_SIZE),
            seed = 69420,
            validation_split = 0.30,
            batch_size = BATCH_SIZE,
            subset = 'both',
        )

        print(f"\nCustom dataset {DS_PATH} provided, loaded successfully")
        print(f"\nDataset images have been resized to ({IMG_SIZE}, {IMG_SIZE})")

        NUM_CLASSES = len(ds_train.class_names)
        label_info = tfds.features.ClassLabel(num_classes=int(NUM_CLASSES))

        # augmentation--err, supposed to be augmentation
        ds_train = ds_train.map(input_preprocess_custom_train, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        print(f"\nNo dataset provided; using standard dataset '{DEFAULT_DATASET_NAME}'")
        (ds_train, ds_test), ds_info = tfds.load(
                DEFAULT_DATASET_NAME, split=["train", "test"], with_info=True, as_supervised=True
        )

        NUM_CLASSES = ds_info.features["label"].num_classes
        label_info = ds_info.features["label"]

        size = (IMG_SIZE, IMG_SIZE)
        ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
        ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
        print("\nDataset images have been resized")

        # one-hot encoding + augmentation of train data
        ds_train = ds_train.map(input_preprocess_tfds_train, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

        # needs to be batched here... batching in .fit doesn't work
        ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
        ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print(f"\nDataset has been loaded; contains {NUM_CLASSES} classes")

    # display_data_sample(ds_train, label_info)
    # display_aug_data_sample(ds_train, label_info)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    print("\nDataset has been resized to uniform IMG_SIZE, labels have been put \
into one-hot (categorical) encoding, the dataset has been batched.")

    return ds_train, ds_test, NUM_CLASSES

def create_model(NUM_CLASSES):
    eff_net = EfficientNetB3(
        include_top = True,
        weights = 'imagenet',

        # weights = None,
        # below only applicable if 'weights = None'
        # classes = NUM_CLASSES,
        # input_shape = (IMG_SIZE, IMG_SIZE, 3),
        # pooling=None,
        # classifier_activation='softmax',
    )

    print(f"\nCreated model: {eff_net}\n")

    eff_net.trainable = False

    inputs = keras.Input(shape = (IMG_SIZE, IMG_SIZE, 3))

    outputs = eff_net(inputs)

    outputs = layers.Dense(256, activation = 'relu')(outputs)
    outputs = layers.Dense(64, activation = 'relu')(outputs)
    outputs = layers.Dense(16, activation = 'relu')(outputs)

    outputs = layers.Dense(NUM_CLASSES, activation = 'softmax')(outputs)

    optimizer = optimizers.legacy.Adam(learning_rate = 0.001)
    loss = losses.CategoricalCrossentropy()

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ['accuracy']
    )

    return model

def main():

    if args['dataset_path']:
        ds_train, ds_test, NUM_CLASSES = load_dataset(args['dataset_path'])
    else:
        ds_train, ds_test, NUM_CLASSES = load_dataset()

    if args['resume_checkpoint']:
        model = tf.keras.models.load_model(args['resume_checkpoint'])
        print(f"\nModel at '{args['resume_checkpoint']}' loaded successfully")

    else:
        model = create_model(NUM_CLASSES) # passing in n_classes for sake of readability & reusability

    CHECKPOINT_DIR = 'checkpoints' # was getting UnboundLocalVar error when defining globally
    if args['CHECKPOINT_DIR']:
        CHECKPOINT_DIR = args['CHECKPOINT_DIR']

    elif os.path.isdir(CHECKPOINT_DIR): # to avoid writing over checkpoints that already exist
        CHECKPOINT_DIR = 'checkpoints_' + str(current_milli_time())[10:]

    print(f"\nSaving checkpoints at {CHECKPOINT_DIR}")
    callback = [
        callbacks.ModelCheckpoint(
            filepath = CHECKPOINT_DIR + '/checkpoint_{epoch:02d}',
            # save_best_only = True,
            verbose = 1,
            save_freq = CHKPT_EPOCH_SAVE_FREQ * len(ds_train),
        )
    ]

    model.summary()

    print() # console formatting ;)
    history = model.fit(
        ds_train,
        epochs = EPOCHS,
        # batch_size = BATCH_SIZE, <-- batching done in preprocessing
        verbose = 1,
        validation_data = ds_test,
        callbacks = callback,
    )

    plot_hist(history)

if __name__ == "__main__":
    main()
