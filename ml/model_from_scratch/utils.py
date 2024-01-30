
"""
Author: Victor J.
Description: Misc useful functions used across multiple files, all aggregated
here for cleanliness and reusability's sake
Date: Winter 2023
"""

import time
import numpy as np

def current_milli_time():
    return round(time.time() * 1000)

def rolling_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w