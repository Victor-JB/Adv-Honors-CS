
"""
Author: Victor J.
Description: Transfer learning from TensorFlow EfficientNet image classification model
with custom Krunker dataset :)
Date: Winter 2023
"""

import numpy as np
import tensorflow_datasets as tfds # for testing with Stanford Dog dataset
import tensorflow as tf  # For tf.data
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.losses as losses
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.applications import EfficientNetB3
import os

# IMG_SIZE is determined by EfficientNet model choice; B3, in this case
IMG_SIZE = 300
BATCH_SIZE = 32 # seems to work well, power of 2
EPOCHS = 100
CHECKPOINT_PATH = 'checkpoints'
DATASET_NAME = "defungi"

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

def display_aug_data_sample(ds_train, label_info=None):
    for image, label in ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
            aug_img = np.array(aug_img)
            plt.imshow(aug_img[0].astype("uint8"))
            plt.title("{}".format(format_label(label_info, label)))
            plt.axis("off")

    plt.show()

# One-hot / categorical encoding
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, num_classes)
    return image, label

def input_preprocess_test(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def load_dataset():
    ds_train, ds_test = utils.image_dataset_from_directory(
        DATASET_NAME,
        label_mode = 'categorical',
        seed = 69420,
        validation_split = 0.30,
        subset = 'both',
    )

    num_classes = len(ds_train.class_names)

    print(f"\nDataset has been loaded; contains {num_classes} classes\nThe dataset:\n{ds_train}")

    size = (IMG_SIZE, IMG_SIZE)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

    print("\nDataset images have been resized")

    label_info = ds_train.features["label"]
    # display_data_sample(ds_train, label_info)
    # display_aug_data_sample(ds_train, label_info=label_info)

    ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print("\nDataset has been resized to uniform IMG_SIZE, labels have been put into \
    one-hot (a.k.a. categorical) encoding, the dataset has been batched.")

    return ds_train, ds_test, num_classes

def create_model():
    eff_net = EfficientNetB3(
        include_top = True,
        # weights = None,
        weights = 'imagenet',

        # below only applicable if 'weights = None'
        # classes = num_classes,
        # input_shape = (IMG_SIZE, IMG_SIZE, 3),
        # pooling=None,
        # classifier_activation='softmax',
    )

    print(f"\n{eff_net=}")

    eff_net.trainable = False

    inputs = keras.Input(shape = (IMG_SIZE, IMG_SIZE, 3))

    outputs = eff_net(inputs)
    outputs = layers.Dense(num_classes, activation = 'softmax')(outputs)

    optimizer = optimizers.legacy.Adam(learning_rate = 0.0001)
    loss = losses.CategoricalCrossentropy()

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ['accuracy']
    )

    return model

def main():
    ds_train, ds_test, num_classes = load_dataset()

    if os.path.isdir('checkpoints'):

        most_recent = max([int(dir.split('_')[1]) for dir in os.listdir(CHECKPOINT_PATH)])
        model = tf.keras.models.load_model(f'{CHECKPOINT_PATH}/checkpoints_{most_recent}')

        loss, acc = model.evaluate(ds_test, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    else:

        model = create_model(num_classes)

        callbacks = [
            callbacks.ModelCheckpoint(
                CHECKPOINT_PATH + '/checkpoints_{epoch:02d}',
                verbose = 2,
                save_freq = 4 * len(ds_train),
            )
        ]

        # model.summary()

        hist = model.fit(
            ds_train,
            epochs = EPOCHS,
            verbose = 1,
            validation_data = ds_test,
            callbacks = callbacks,
        )

        plot_hist(hist)

if __name__ == "__main__":
    main()
