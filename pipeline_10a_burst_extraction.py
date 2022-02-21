import sys
import json
import os.path as op
from os import sep
from mne import read_epochs
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import maximum_filter
from utilities import files
from extra.tools import fwhm_burst_norm
from fooof import FOOOF
import pandas as pd
import numpy as np

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
hi_pass = parameters["hi_pass_filter"]
sub_path = op.join(path, "data")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

all_folders = files.get_folders_files(sub_path)[0]
all_folders = [i for i in all_folders if "-epo" in i]
all_folders.sort()

all_visual = [i for i in all_folders if "visual" in i]
all_motor = [i for i in all_folders if "motor" in i]
visual_motor_folders = list(zip(all_visual, all_motor))

# getting channel info from epoch file
epo_path = files.get_files(sub_path, "sub-", "-epo.fif")[2][0]
info = read_epochs(epo_path, verbose=False)
info.pick_types(meg=True, ref_meg=False, misc=False)
info = info.info

# info
freqs = np.linspace(1,120, num=400)
beta_range = np.where((freqs >= 13) & (freqs <= 30))[0]
vis_time = np.linspace(-1, 2, num=1801)[75:-75]
mot_time = np.linspace(-1, 1.5, num=1501)[75:-75]

bursts_vis = {i: [] for i in info.ch_names}
bursts_mot = {i: [] for i in info.ch_names}

for block, (vis_folder, mot_folder) in enumerate(visual_motor_folders):
    vis_name = vis_folder.split(sep)[-1]
    mot_name = mot_folder.split(sep)[-1]
    print(vis_name, mot_name)
    vis_files = files.get_files(vis_folder, "", ".npy")[2]
    vis_files.sort()
    mot_files = files.get_files(mot_folder, "", ".npy")[2]
    mot_files.sort()

    TF_vis = {i: [] for i in info.ch_names}
    psd_vis = {i: [] for i in info.ch_names}
    TF_mot = {i: [] for i in info.ch_names}
    threshold = {}
    

    for trial, vis_file in enumerate(vis_files):
        print(vis_file.split(sep)[-1], block, trial, block*56+trial)
        data = np.load(vis_file)
        for ch_ix, ch_name in enumerate(info.ch_names):
            TF = data[ch_ix, beta_range, 75:-75]
            TF_vis[ch_name].append([block, trial, TF])
            psd = np.mean(data[ch_ix, :, 75:-75], axis=1)
            psd_vis[ch_name].append([block, trial, psd])
    
    for ch_name in info.ch_names:
        psd_mean_vis = [k for i,j,k in psd_vis[ch_name]]
        psd_mean_vis = np.vstack(psd_mean_vis)
        psd_mean_vis = np.mean(psd_mean_vis, axis=0)
        ff = FOOOF()
        ff.fit(freqs, psd_mean_vis, [1, 120])
        threshold[ch_name] = 10**ff._ap_fit[beta_range]

    for trial, mot_file in enumerate(mot_files):
        print(mot_file.split(sep)[-1], block, trial, block*56+trial)
        data = np.load(mot_file)
        for ch_ix, ch_name in enumerate(info.ch_names):
            TF = data[ch_ix, beta_range, 75:-75]
            TF_mot[ch_name].append([block, trial, TF])
    

    for ch_name in info.ch_names:
        for block, trial, TF in TF_vis[ch_name]:
            vis_TF = TF - threshold[ch_name].reshape(-1,1)
            vis_TF[vis_TF<=0] = 0
            mm_map_v = minmax_scale(vis_TF) - 0.5
            mm_map_v[mm_map_v<=0] = 0
            mm_map_v = minmax_scale(mm_map_v)
            vis_max = maximum_filter(vis_TF, size=(25, 200), mode="nearest")
            vis_max = vis_TF * mm_map_v == vis_max
            x, y = np.where(vis_max == 1)
            pks = list(zip(x, y))
            for pk in pks:
                right_loc, left_loc, up_loc, down_loc = fwhm_burst_norm(vis_TF * mm_map_v, pk)
                if 0 not in (right_loc + left_loc, up_loc + down_loc):
                    bursts_vis[ch_name].append([
                        int(pk[0]), 
                        int(pk[1]+75), 
                        int(right_loc), 
                        int(left_loc), 
                        int(up_loc), 
                        int(down_loc), 
                        float(vis_TF[pk]), 
                        int(trial),
                        int(block),
                        int(block*56+trial)
                    ])

    for ch_name in info.ch_names:
        for block, trial, TF in TF_mot[ch_name]:
            mot_TF = TF - threshold[ch_name].reshape(-1,1)
            mot_TF[mot_TF<=0] = 0
            mm_map_m = minmax_scale(mot_TF) - 0.5
            mm_map_m[mm_map_m<=0] = 0
            mm_map_m = minmax_scale(mm_map_m)
            mot_max = maximum_filter(mot_TF, size=(25, 200), mode="nearest")
            mot_max = mot_TF * mm_map_m == mot_max
            x, y = np.where(mot_max == 1)
            pks = list(zip(x, y))
            for pk in pks:
                right_loc, left_loc, up_loc, down_loc = fwhm_burst_norm(mot_TF * mm_map_m, pk)
                if 0 not in (right_loc + left_loc, up_loc + down_loc):
                    bursts_mot[ch_name].append([
                        int(pk[0]), 
                        int(pk[1]+75), 
                        int(right_loc), 
                        int(left_loc), 
                        int(up_loc), 
                        int(down_loc), 
                        float(mot_TF[pk]), 
                        int(trial),
                        int(block),
                        int(block*56+trial)
                    ])
    
vis_json_name = "{}-visual-burst.json".format(subject_id)
mot_json_name = "{}-motor-burst.json".format(subject_id)

vis_json_path = op.join(subject, vis_json_name)
mot_json_path = op.join(subject, mot_json_name)

with open(vis_json_path, "w") as fp:
    json.dump(bursts_vis, fp, indent=4)
print("SAVED", vis_json_path)

with open(mot_json_path, "w") as fp:
    json.dump(bursts_mot, fp, indent=4)
print("SAVED", mot_json_path)