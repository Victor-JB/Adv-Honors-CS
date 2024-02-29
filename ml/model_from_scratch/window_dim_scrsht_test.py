
"""
Description: Nice little gui for entering screenshotter dimensions and seeing what
portion of the screen is captured; is nice for fine-tuning the dimensions you want
to screenshot
Author: Victor J.
Date: Feb 2024
"""

import mss.tools
import cv2
import tkinter as tk
import numpy as np

screenshotter = mss.mss()

root = tk.Tk()
root.geometry("600x400")

# tk variables for storing ints recieved from Entrys
top_left_x = tk.IntVar()
top_left_y = tk.IntVar()

width_var = tk.IntVar()
height_var = tk.IntVar()

# ---------------------------------------------------------------------------- #
def gen_test_img():
    top_x = top_left_x.get()
    top_y = top_left_y.get()

    width = width_var.get()
    height = height_var.get()

    bounds = {'top': top_y, 'left': top_x, 'width': width, 'height': height}

    img = np.array(screenshotter.grab(bounds))

    cv2.imshow(f"Window dimensions of ({top_x}, {top_y}), width={width}, height={height}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #

top_x_lbl = tk.Label(root, text = 'Top Left X Coord', font=('calibre',10, 'bold'))
top_x_entry = tk.Entry(root, textvariable = top_left_x, font=('calibre',10,'normal'))

top_y_lbl = tk.Label(root, text = 'Top Left Y Coord', font = ('calibre',10,'bold'))
top_y_entry = tk.Entry(root, textvariable = top_left_y, font = ('calibre',10,'normal'))

width_lbl = tk.Label(root, text = 'Window Width', font = ('calibre',10,'bold'))
width_entry = tk.Entry(root, textvariable = width_var, font = ('calibre',10,'normal'))

height_lbl = tk.Label(root, text = 'Window Height', font = ('calibre',10,'bold'))
height_entry = tk.Entry(root, textvariable = height_var, font = ('calibre',10,'normal'))

gen_img_button = tk.Button(root,text = 'Submit', command = gen_test_img)

# placing the label and entry in
# the required position using grid
# method
top_x_lbl.grid(row=0,column=0)
top_x_entry.grid(row=0,column=1)

top_y_lbl.grid(row=1,column=0)
top_y_entry.grid(row=1,column=1)

width_lbl.grid(row=2,column=0)
width_entry.grid(row=2,column=1)

height_lbl.grid(row=3,column=0)
height_entry.grid(row=3,column=1)

gen_img_button.grid(row=4,column=1)

# performing an infinite loop
# for the window to display
root.mainloop()
