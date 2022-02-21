import json
from collections import OrderedDict
import numpy as np
from scipy.interpolate import interp1d

def update_key_value(file, key, value):
    """
    Function to update a key value in a JSON file. If passed
    key is not in the JSON file, it is going to be appended 
    at the end of the file.
    """
    with open(file, "r") as json_file:
        data = json.load(json_file, object_pairs_hook=OrderedDict)
        data[key] = value
    
    with open(file, "w") as json_file:
        json.dump(data, json_file, indent=4)


def dump_the_dict(file, dictionary):
    """
    Function dumps dictionary to a JSON file.
    """
    with open(file, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


def resamp_interp(x, y, new_x):
    """
    returns resampled an interpolated data
    """
    resamp = interp1d(x, y, kind='linear', fill_value='extrapolate')
    new_data = resamp(new_x)
    return new_data


def cart2pol(x, y):
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return [radius, angle]


def pol2cart(radius, angle):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return [x, y]


def fwhm_burst_norm(TF, peak):
    right = TF[peak[0], peak[1]:]
    try:
        right_loc = np.where(right <= TF[peak]/2)[0][0]
    except:
        if np.average(right <= TF[peak]/2) == 0 or np.average(right <= TF[peak]/2) == np.nan:
            right_loc = right.shape[0]-1
        else:
            right_loc = 0
    up = TF[peak[0]:, peak[1]]
    try:
        up_loc = np.where(up <= TF[peak]/2)[0][0]
    except:
        if np.average(up <= TF[peak]/2) == 0 or np.average(up <= TF[peak]/2) == np.nan:
            up_loc = up.shape[0]-1
        else:
            up_loc = 0
    left = TF[peak[0], :peak[1]]
    try:
        left_loc = np.where(np.flip(left) <= TF[peak]/2)[0][0]
    except:
        if np.average(left <= TF[peak]/2) == 0 or np.average(left <= TF[peak]/2) == np.nan:
            left_loc = left.shape[0]-1
        else:
            left_loc = 0
    down = TF[:peak[0], peak[1]]
    try:
        down_loc = np.where(np.flip(down) <= TF[peak]/2)[0][0]
    except:
        if np.average(down <= TF[peak]/2) == 0 or np.average(down <= TF[peak]/2) == np.nan:
            down_loc = down.shape[0]-1
        else:
            down_loc = 0
    horiz = np.min([left_loc, right_loc])
    vert = np.min([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert
    return right_loc, left_loc, up_loc, down_loc