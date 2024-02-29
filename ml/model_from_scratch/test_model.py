
"""
Author: Victor J.
Description: Script for testing and evaluating a given efficient_net model
Date: Winter 2023
"""

import os
import tensorflow as tf
import cv2
import argparse
from utils import load_dataset
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint_path", required = True,
               help = "Checkpoint path from which to load model")
ap.add_argument("-i", "--test_image_path", required = False,
               help = "Image path from which to test with one image")
ap.add_argument("-d", "--test_image_dir", required = False,
               help = "Image path from which to test with one image")
ap.add_argument("-e", "--eval_ds_path", required = False,
               help = "Path with dataset from which to evaluate the model")
ap.add_argument("-t",
                "--model_type",
                required = False,
                help = "Specify which model you want to train with (0 or 1; will change to v1 or v2)",
                default = 1,
)
args = vars(ap.parse_args())

if int(args['model_type']) == 0:
    IMG_SIZE = 227
elif int(args['model_type']) == 1:
    IMG_SIZE = 302
else:
    raise ValueError('"--model_type" argument provided is not int 0 or 1')

IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

def main():

    print(f"Loading model from provided checkpoint path '{args['checkpoint_path']}'...")
    # compile False loads in the model just for inference
    model = tf.keras.models.load_model(args['checkpoint_path'], compile=False)

    print(f"\nModel at '{args['checkpoint_path']}' loaded successfully\n")
    model.summary()

    if args['test_image_path']:
        print(f"\nTesting with custom image path '{args['test_image_path']}'")
        img = cv2.imread(args['test_image_path'])
        img = cv2.resize(img, IMG_SHAPE) # resize image to match model's expected sizing
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        result = model.predict(img)
        classes = result.argmax(axis=-1)
        print(classes)

    if args['test_image_dir']:
        print(f"\nTesting with directory of images at '{args['test_image_dir']}'")
        images = os.listdir(args['test_image_dir'])
        for image in images:
            img_path = args['test_image_dir'].strip('/') + "/" + image
            img = cv2.imread(img_path)
            print(img_path)
            img = cv2.resize(img, IMG_SHAPE) # resize image to match model's expected sizing
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

            result = model.predict(img)
            classes = result.argmax(axis=-1)
            print(classes)

    if args['eval_ds_path']:
        print(f"\n\nEvaluating it with dataset at {args['eval_ds_path']}")

        _, ds_test, NUM_CLASSES = load_dataset(args['eval_ds_path'], IMG_SIZE, 32)

        optimizer = optimizers.Adam(learning_rate=0.0001)
        loss = losses.CategoricalCrossentropy()

        model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = ['accuracy'],
        )
        loss, acc = model.evaluate(ds_test, verbose=1)

        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    main()
