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