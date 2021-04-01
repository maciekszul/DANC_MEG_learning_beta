import sys
import json
import numpy as np
import pandas as pd
import mne
import os.path as op
from utilities import files
import matplotlib.pylab as plt

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    file_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

try:
    json_file = sys.argv[3]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]
sub_path = op.join(path, "data")
der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(sub_path)[0]
subject = subjects[index]
subject_id = subject.split("/")[-1]

meg_path = op.join(subject, "ses-01", "meg")

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
raw_paths.sort()
