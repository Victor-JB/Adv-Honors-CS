
"""
Author: Victor J.
Description: Creating the krunker player dataset for my model
Date: Winter 2023
"""

import mss.tools
from pynput import keyboard
import argparse
import numpy as np
from PIL import Image
import time

try:
    import pydirectinput as pymouseutil # For Windows
except AttributeError:
    import pyautogui as pymouseutil     # For Mac

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--create_ds", required = False,
               help = "Toggle whether dataset creation is on or off",
               action="store_true") # no input is required for argument; just the flag
args = vars(ap.parse_args())

SCRNSHOT_PLAYER = False
SCRNSHOT_NOT_PLAYER = False

screenshotter = mss.mss()

# needs to be defined up here for keyboard.Listener(on_release)
def on_release(key):
    key = str(key).strip("'")

    if key == 'p':
        global SCRNSHOT_PLAYER
        SCRNSHOT_PLAYER = not SCRNSHOT_PLAYER
        SCRNSHOT_NOT_PLAYER = False

    if key == 'n':
        SCRNSHOT_NOT_PLAYER = not SCRNSHOT_NOT_PLAYER
        SCRNSHOT_PLAYER = False

    return

def current_milli_time():
    return round(time.time() * 1000)

listener = keyboard.Listener(on_release=on_release)
listener.start()

def main():

    # Automatically get screen size and set screenshot dimensions
    screen_width, screen_height = pymouseutil.size()
    screenshotter_bounding_box = {'top': 0, 'left': 0,
                                  'width': screen_width,
                                  'height': screen_height}

    while True:
        if SCRNSHOT_PLAYER:
            sct_img = screenshotter.grab(screenshotter_bounding_box)

            output = f"krunker_img/PLAYER/PLAYER_{current_milli_time()}.jpeg"

            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

        elif SCRNSHOT_NOT_PLAYER:
            sct_img = screenshotter.grab(screenshotter_bounding_box)

            now = datetime.datetime.now().strftime("%H%M%S")
            output = f"krunker_img/NO_PLAYER/NO_PLAYER_{current_milli_time()}.jpeg"

            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

if __name__ == "__main__":
    main()
