
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
from utils import current_hr_time

try:
    import pydirectinput as pymouseutil # For Windows
except AttributeError:
    import pyautogui as pymouseutil     # For Mac

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--model_path", required = True,
               help = "Enter model ckpt path from which to load model")
ap.add_argument("-t",
                "--model_type",
                required = False,
                help = "Specify which model you want to train with (0 or 1; will change to v1 or v2, respectively)",
                default = 1,
)
args = vars(ap.parse_args())

DETECTION_ACTIVE = False
CAPTURE_VIDEO = False
ON = True
FRAMES_ARR  = []

if int(args['model_type']) == 0:
    IMG_SIZE = 227
elif int(args['model_type']) == 1:
    IMG_SIZE = 302
else:
    raise ValueError('"--model_type" argument provided is not int 0 or 1')

IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

screenshotter = mss.mss()

# needs to be defined up here for keyboard.Listener(on_release)
def on_release(key):
    key = str(key).strip("'")

    global DETECTION_ACTIVE
    global CAPTURE_VIDEO
    global ON

    if key == 'p':
        DETECTION_ACTIVE = not DETECTION_ACTIVE
        print("\n\nDETECTION is", DETECTION_ACTIVE)

    elif key == 'v':
        CAPTURE_VIDEO = not CAPTURE_VIDEO
        print("\n\CAPTURE_VIDEO is", CAPTURE_VIDEO)

    elif key == 'q':
        ON = False

    return

def load_model():
    print(f"\nLoading model from provided checkpoint path '{args['model_path']}'...")
    model = tf.keras.models.load_model(args['model_path'])

    model.summary()
    print(f"Model at '{args['model_path']}' loaded successfully\n")

    return model

listener = keyboard.Listener(on_release=on_release)
listener.start()

def main():

    # Automatically get screen size and set screenshot dimensions
    screen_width, screen_height = pymouseutil.size()
    """
    When split screen on my windows monitor
    """
    screenshotter_bounding_box = {
        'top': 0, 
        'left': 0,
        'width': 1950,
        'height': 870,
    }

    """
    screenshotter_bounding_box = {
        'top': 0, 
        'left': 0,
        'width': screen_width,
        'height': screen_height,
    }
    """

    print("Keyboard being input read... ready to record screen in real time\n1. 'p' key \
to toggle detection\n2. 'v' key to toggle video saving\n3. 'q' key to quit the program")

    model = load_model()

    print("Ready for detection...")
    while ON:
        if DETECTION_ACTIVE:
            sct_img = np.array(screenshotter.grab(screenshotter_bounding_box))

            # converting mss grab to proper rgb format and such
            im = np.flip(sct_img[:, :, :3], 2)  # 1
            cv2_compat_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2

            if CAPTURE_VIDEO:
                FRAMES_ARR.append(sct_img)

            img = cv2.resize(cv2_compat_im, IMG_SHAPE) # resize image to match model's expected sizing
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

            result = model.predict(img)
            predict_class = result.argmax(axis=-1)

            if predict_class == 0:
                print("NO PLAYER DETECTED")
            else:
                print("PLAYER DETECTED!")
    
    if len(FRAMES_ARR) > 0:
        x = FRAMES_ARR[0].shape[1]
        y = FRAMES_ARR[0].shape[0]
        # print("printing shape of single frame...", FRAMES_ARR[0].shape, 'x and y:', x, y)
        out = cv2.VideoWriter(f"krunker_{current_hr_time()}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (x, y))
        for frame in FRAMES_ARR:
            out.write(frame) # frame is a numpy.ndarray with shape (screen_height, screen_width, 4)
        out.release()


if __name__ == "__main__":
    main()
