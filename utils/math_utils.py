import numpy as np


def fix_angle(a, offset=0):
    return (a - offset) % 360 + offset


def my_arctan2(x, y, offset=-180):
    return fix_angle(np.rad2deg(np.arctan2(y, x)), offset)
