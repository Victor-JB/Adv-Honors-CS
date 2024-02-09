
"""
Author: Victor J.
Description: Will screen record and predict if players are on screen in real time
Date: Winter 2024
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disables extensive tensorflow debugging msgs

import mss.tools
from pynput import keyboard
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import argparse

try:
    import pydirectinput as pymouseutil # For Windows
except AttributeError:
    import pyautogui as pymouseutil     # For Mac

DETECTION_ACTIVE = False
ON = True

IMG_SIZE = 227
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--model_path", required = True,
               help = "Enter model ckpt path from which to load model")
args = vars(ap.parse_args())

screenshotter = mss.mss()

# needs to be defined up here for keyboard.Listener(on_release)
def on_release(key):
    key = str(key).strip("'")

    global DETECTION_ACTIVE

    if key == 'p':
        DETECTION_ACTIVE = not DETECTION_ACTIVE
        print("\n\nDETECTION is", DETECTION_ACTIVE)

    if key == 'q':
        global ON
        ON = False

    return

def load_model():
    print(f"\nLoading model from provided checkpoint path '{args['model_path']}'...")
    model = tf.keras.models.load_model(args['model_path'])

    print(f"Model at '{args['model_path']}' loaded successfully\n")
    model.summary()

    return model

listener = keyboard.Listener(on_release=on_release)
listener.start()

def main():

    # Automatically get screen size and set screenshot dimensions
    screen_width, screen_height = pymouseutil.size()
    screenshotter_bounding_box = {'top': 0, 'left': 0,
                                  'width': screen_width,
                                  'height': screen_height}

    print("Keyboard being input read... ready to record screen in real time\n1. 'p' key \
to toggle detection\n2. 'q' key to quit the program")

    model = load_model()

    print("\nReady for detection...")
    while ON:
        if DETECTION_ACTIVE:
            sct_img = np.array(screenshotter.grab(screenshotter_bounding_box))

            # converting mss grab to proper rgb format and such
            im = np.flip(sct_img[:, :, :3], 2)  # 1
            cv2_compat_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2

            img = cv2.resize(cv2_compat_im, IMG_SHAPE) # resize image to match model's expected sizing
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

            result = model.predict(img)
            predict_class = result.argmax(axis=-1)

            print(predict_class)

            if predict_class == 0:
                print("NO PLAYER DETECTED")
            else:
                print("PLAYER DETECTED!")



if __name__ == "__main__":
    main()
