import json
import math
import copy
import sys
import warnings
from os import sep
import os.path as op
import numpy as np
from fooof import FOOOF
from extra.tools import extract_bursts
from mne import read_epochs
from utilities import files


warnings.filterwarnings("ignore")
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

#setting the paths and extracting files
slt_mot_paths  = [i for i in files.get_folders_files(sub_path)[0] if "motor" in i]
slt_vis_paths = [i for i in files.get_folders_files(sub_path)[0] if "visual" in i]
epo_mot_paths  = files.get_files(sub_path, "sub", "motor-epo.fif")[2]
epo_vis_paths = files.get_files(sub_path, "sub", "visual-epo.fif")[2]
beh_match_path = files.get_files(sub_path, "sub", "beh-match.json")[2][0]
with open(beh_match_path) as f:
    beh_match = json.load(f)
slt_mot_paths.sort()
slt_vis_paths.sort()
epo_mot_paths.sort()
epo_vis_paths.sort()
epo_slt_mot_vis = list(zip(epo_mot_paths, epo_vis_paths, slt_mot_paths, slt_vis_paths))

info = read_epochs(epo_mot_paths[0], verbose=False)
info.pick_types(meg=True, ref_meg=False, misc=False)
info = info.info
freqs = np.linspace(1,120, num=400)
search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
beta_lims = [13, 30]

vis_output = {}
mot_output = {}

# for ch_ix, channel in enumerate ([info.ch_names[10]]):
for ch_ix, channel in enumerate(info.ch_names):
    vis_blocks = {}
    mot_blocks = {}
    # for block, (epo_mot_p, epo_vis_p, slt_mot_p, slt_vis_p) in enumerate(epo_slt_mot_vis[:2]):
    for block, (epo_mot_p, epo_vis_p, slt_mot_p, slt_vis_p) in enumerate(epo_slt_mot_vis):
        beh_match_vis = beh_match[epo_vis_p.split(sep)[-1]]
        beh_match_mot = beh_match[epo_mot_p.split(sep)[-1]]
        slt_vis_nps = files.get_files(slt_vis_p, "", ".npy")[2]
        slt_vis_nps.sort()
        slt_mot_nps = files.get_files(slt_mot_p, "", ".npy")[2]
        slt_mot_nps.sort()
        epo_vis = read_epochs(epo_vis_p, verbose=False)
        epo_vis = epo_vis.pick_types(meg=True, ref_meg=False, misc=False)
        epo_mot = read_epochs(epo_mot_p, verbose=False)
        epo_mot = epo_mot.pick_types(meg=True, ref_meg=False, misc=False)
        print("start:", "{}/274".format(ch_ix+1), subject_id, block, "vis, beh_match {}, nps {}, epo {}".format(len(beh_match_vis), len(slt_vis_nps), len(epo_vis)))
        print("start:", "{}/274".format(ch_ix+1), subject_id, block, "mot, beh_match {}, nps {}, epo {}".format(len(beh_match_mot), len(slt_mot_nps), len(epo_mot)))
        vis_TF = []
        vis_psd = []
        for vis_p in slt_vis_nps:
            data = np.load(vis_p)
            TF = data[ch_ix, search_range, :]
            vis_TF.append(TF)
            psd = np.mean(data[ch_ix, :, :], axis=1)
            vis_psd.append(psd)
            del(data)

        mot_TF = []
        mot_psd = []
        for mot_p in slt_mot_nps:
            data = np.load(mot_p)
            TF = data[ch_ix, search_range, :]
            mot_TF.append(TF)
            psd = np.mean(data[ch_ix, :, :], axis=1)
            mot_psd.append(psd)
            del(data)

        vis_psd_avg = np.mean(np.vstack(vis_psd), axis=0)
        mot_psd_avg = np.mean(np.vstack(mot_psd), axis=0)

        ff_vis = FOOOF()
        ff_vis.fit(freqs, vis_psd_avg, [1, 120])
        ap_fit_v = 10 ** ff_vis._ap_fit

        vis_raw = epo_vis.get_data()[:,ch_ix,:]
        mot_raw = epo_mot.get_data()[:,ch_ix,:]
        fooof_thresh = ap_fit_v[search_range].reshape(-1, 1)
        sfreq = epo_vis.info['sfreq']
        block_vis_burst = extract_bursts(
            vis_raw, 
            vis_TF, 
            epo_vis.times, 
            freqs[search_range], 
            beta_lims, 
            fooof_thresh, 
            sfreq,
            beh_ix=beh_match_vis
        )
        vis_blocks[block] = block_vis_burst
        
        block_mot_burst = extract_bursts(
            mot_raw, 
            mot_TF, 
            epo_mot.times, 
            freqs[search_range], 
            beta_lims, 
            fooof_thresh, 
            sfreq,
            beh_ix=beh_match_mot
        )
        mot_blocks[block] = block_mot_burst
    
    # most of the nonsense here is to clean up the data type for savin in JSON
    # appending all blocks together to avoid too nested JSON file
    vis_results = {i:[] for i in vis_blocks[0].keys()}
    vis_results["block"] = []
    vis_results["pp_ix"] = []
    for bl in vis_blocks.keys():
        for key in vis_blocks[bl]:
            if key == "waveform":
                vis_results[key].extend(vis_blocks[bl][key].astype(float).tolist())
            elif key == "trial":
                vis_results[key].extend(vis_blocks[bl][key].astype(int).tolist())
            else:
                vis_results[key].extend(np.array(vis_blocks[bl][key]).astype(float).tolist())
        vis_results["block"].extend(np.tile(bl, len(vis_blocks[bl]["trial"])).astype(int).tolist())
        vis_results["pp_ix"].extend((bl*56 + np.array(vis_blocks[bl]["trial"])).astype(int).tolist())
    vis_output[channel] = vis_results
    
    mot_results = {i:[] for i in mot_blocks[0].keys()}
    mot_results["block"] = []
    mot_results["pp_ix"] = []
    for bl in mot_blocks.keys():
        for key in mot_blocks[bl]:
            if key == "waveform":
                mot_results[key].extend(mot_blocks[bl][key].astype(float).tolist())
            elif key == "trial":
                mot_results[key].extend(mot_blocks[bl][key].astype(int).tolist())
            else:
                mot_results[key].extend(mot_blocks[bl][key])
        mot_results["block"].extend(np.tile(bl, len(mot_blocks[bl]["trial"])).astype(int).tolist())
        mot_results["pp_ix"].extend((bl*56 + np.array(mot_blocks[bl]["trial"])).astype(int).tolist())
    mot_output[channel] = mot_results


vis_json_name = "{}-visual-burst-iter.json".format(subject_id)
mot_json_name = "{}-motor-burst-iter.json".format(subject_id)

vis_json_path = op.join(subject, vis_json_name)
mot_json_path = op.join(subject, mot_json_name)

with open(vis_json_path, "w") as fp:
    json.dump(vis_output, fp, indent=4)
print("SAVED", subject_id, vis_json_path)

with open(mot_json_path, "w") as fp:
    json.dump(mot_output, fp, indent=4)
print("SAVED", subject_id, mot_json_path)