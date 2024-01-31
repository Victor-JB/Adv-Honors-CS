
"""
Author: Victor J.
Description: Script for testing and evaluating a given efficient_net model
Date: Winter 2023
"""

import tensorflow as tf
import cv2
# from model_function import IMG_SIZE
import argparse
# from model_function import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint_path", required = False,
               help = "Checkpoint path from which to load model")
ap.add_argument("-i", "--test_image_path", required = False,
               help = "Image path from which to test with one image")

# BELOW BROKEN: can't import efficient_net without argparse stuff being overwritten
# by the other file (efficient_train) overwriting argparse variables of this file; strange
ap.add_argument("-e", "--eval_ds_path", required = False,
               help = "Path with dataset from which to evaluate the model")
args = vars(ap.parse_args())

IMG_SIZE = 227
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
DEFAULT_CKP_PATH = "checkpoints_255/checkpoint_40"
DEF_TEST_IMG_PATH = "new_test_krunker_images"

def main():

    if args['checkpoint_path']:
        print(f"Loading model from provided checkpoint path '{args['checkpoint_path']}'...")
        model = tf.keras.models.load_model(args['checkpoint_path'])

        print(f"\nModel at '{args['checkpoint_path']}' loaded successfully\n")
        model.summary()

    else:
        print(f"Loading model from default checkpoint path '{DEFAULT_CKP_PATH}'...")
        model = tf.keras.models.load_model(DEFAULT_CKP_PATH)

        print(f"\nModel at '{DEFAULT_CKP_PATH}' loaded successfully\n")
        model.summary()

    if args['test_image_path']:
        print(f"\nTesting with custom image path '{args['test_image_path']}'")

        img = cv2.imread(args['test_image_path'])
        img = cv2.resize(img, IMG_SHAPE) # resize image to match model's expected sizing
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        result = model.predict(img)
        classes = result.argmax(axis=-1)
        print(classes)

    else:
        print(f"\nTesting with default image path '{DEF_TEST_IMG_PATH}'")

        img = cv2.imread(DEF_TEST_IMG_PATH)
        img = cv2.resize(img, IMG_SHAPE) # resize image to match model's expected sizing
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        result = model.predict(img)
        classes = result.argmax(axis=-1)
        print(classes)

    if args['eval_ds_path']:
        print(f"\n\nEvaluating it with dataset at {args['eval_ds_path']}")
        _, ds_test, NUM_CLASSES = load_dataset(args['eval_ds_path'])
        loss, acc = model.evaluate(ds_test, verbose=1)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    main()
