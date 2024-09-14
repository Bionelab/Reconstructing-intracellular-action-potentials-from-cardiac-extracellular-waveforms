import os
import pickle
import numpy as np
# from utils_gen import*
# from load_recording_data import *
seed=42
data_loc = os. path.join(os.path.dirname(os.getcwd()),'general/eap_iap_data.pkl')
with open(data_loc, 'rb') as inp:
  all_recordings = pickle.load(inp)
data={}
all_data_keys = all_recordings.keys()
for key in all_data_keys:    
    sn,intras,extras,intras_normalized,extras_normalized = all_recordings[key]
    mins = extras_normalized[:, 4000:5000].min(axis=1) > -0.3
    sn,intras,extras,intras_normalized,extras_normalized = sn[mins],intras[mins],extras[mins],intras_normalized[mins],extras_normalized[mins]

    intras_normalized2 = intras_normalized+0.11
    intras_normalized2 = intras_normalized2/1.1
    extras_normalized2 = extras_normalized - extras_normalized[:, 0][:, np.newaxis]
  

    if len (intras) > 10:
        data[(key,'extra_norm1')] = extras_normalized[10:-5]
        data[(key,'extra')]=extras_normalized2[10:-5]
        data[(key,'intra')]=intras_normalized2[10:-5]
        data[(key,'extra_raw')]=extras[10:-5]
        data[(key,'intra_raw')]=intras[10:-5]
        data[(key,'s/n')]=sn[10:-5]

del all_recordings
