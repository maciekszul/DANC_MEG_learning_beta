import os
import sys
import json
import os.path as op
import subprocess as sp
from utilities import files
import matlab.engine

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)


path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]
print(subject)

raw_meg_path = op.join(path, "data", subject_id, "ses-01", "meg")
ds_paths = files.get_folders_files(raw_meg_path)[0]
ds_paths = [i for i in ds_paths if "misc" not in i]
ds_paths.sort()
res4_paths = [files.get_files(i, "", ".res4")[2][0] for i in ds_paths]
print(res4_paths)

fif_paths = files.get_files(subject, "sub", "-epo.fif")[2]
fif_paths.sort()

# pial, mri, nas, lpa, rpa