
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

SCRNSHOT_PLAYER = False
SCRNSHOT_NOT_PLAYER = False
ON = True

screenshotter = mss.mss()

# needs to be defined up here for keyboard.Listener(on_release)
def on_release(key):
    key = str(key).strip("'")

    global SCRNSHOT_PLAYER
    global SCRNSHOT_NOT_PLAYER

    if key == 'p':
        SCRNSHOT_PLAYER = not SCRNSHOT_PLAYER
        SCRNSHOT_NOT_PLAYER = False
        print("\n\nSCRNSHOT_PLAYER is", SCRNSHOT_PLAYER)
        print("SCRNSHOT_NOT_PLAYER is", SCRNSHOT_NOT_PLAYER)

    if key == 'n':
        SCRNSHOT_NOT_PLAYER = not SCRNSHOT_NOT_PLAYER
        SCRNSHOT_PLAYER = False
        print("\n\nSCRNSHOT_NOT_PLAYER is", SCRNSHOT_NOT_PLAYER)
        print("SCRNSHOT_PLAYER is", SCRNSHOT_PLAYER)

    if key == 'q':
        global ON
        ON = False

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

    print("Keyboard being input read... ready to take screenshots\n1. 'p' key \
to toggle screenshotting when player is IN the frame\n2. 'n' key to toggle when \
player is NOT in view")

    while ON:
        if SCRNSHOT_PLAYER:
            sct_img = screenshotter.grab(screenshotter_bounding_box)

            output = f"krunker_img/PLAYER/PLAYER_{current_milli_time()}.jpeg"

            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

        elif SCRNSHOT_NOT_PLAYER:
            sct_img = screenshotter.grab(screenshotter_bounding_box)

            output = f"krunker_img/NO_PLAYER/NO_PLAYER_{current_milli_time()}.jpeg"

            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

if __name__ == "__main__":
    main()
