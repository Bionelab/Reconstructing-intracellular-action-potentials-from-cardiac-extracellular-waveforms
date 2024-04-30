
# pip install McsPyDataTools

# from google.colab import drive
# drive.mount('/content/drive')

# %cd 'NEA AI_Copy of Stanford Files'
# %ls


# Commented out IPython magic to ensure Python compatibility.
# %cd 'drive/Shareddrives/NanoEngineering - BioNE Lab  /Ongoing Projects/NEA AI_Copy of Stanford Files'

# Commented out IPython magic to ensure Python compatibility.
import sys
import peak_finder
from peak_finder import butter_bandpass_filter, get_peak_indices, find_poration_pulses
import os
import pickle
import numpy as np
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,peak_widths
from sklearn.model_selection import KFold, train_test_split
# %matplotlib inline
from pandas import DataFrame as df
import math
fs = 5000


#####

lowcut_iap_sn = 0.125
highcut_iap_sn  = fs/32
order_sn = 3
# f_intra1 = butter_bandpass_filter(intra1, lowcut_iap_sn, highcut_iap_sn, fs,order=order_sn) # filtered singal
# n_intra1 = butter_bandpass_filter(intra1,highcut_iap_sn+0.001,fs/2.-1,fs, order=order_sn)#get high frequency component noise 
# spike1a = f_intra1[t_p-1000:t_p+7000]
# noise1a = n_intra1[t_p-1000:t_p+7000]
# sp_a,npower_a,sn_a = singal_noise_power(spike1a,noise1a)
############

# from recording, returns the whole recording, all the cannels, all the filtered singals, filtered signals for s/n and filtered noises for noise, and the positions for pilse

def singal_noise_power(spike, noise):

    sp_ = np.mean(spike ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(noise ** 2)  # Square of the signal's amplitude
    sn_ = 10 * math.log10(sp_ / npower_)
    return sp_,npower_,sn_


def get_paired_dat_and_info(recording, pair):
    file=h5.File(recording,'r')
    file=file['Data']['Recording_0']['AnalogStream']['Stream_0']
    channel_to_elec=np.asarray([(el[0],int(el[4])) for el in list(file['InfoChannel'])])
    data=np.asarray(file['ChannelData'])
    filtered_data=butter_bandpass_filter(data,1,fs/2.-1,fs)
    p_idx=find_poration_pulses(filtered_data)
    #are there poration pulses ? Then make a slice to split data
    if len(p_idx)>0:
        poration_splits=np.hstack(([0],p_idx,len(filtered_data[0])))
    else:
        poration_splits=[]
    n_pts=len(filtered_data[0])
    duration=n_pts//fs
    return data,channel_to_elec,filtered_data,poration_splits
    
def get_peaks_from_both_channels(pair,data,filtered_data,channel_to_elec,poration_splits,buffer=50000):
    peaks=[]
    noise_levels=[]
    c_idx=[]
    for idx in pair:
        i=channel_to_elec[np.asarray(channel_to_elec)[:,1]==idx][0][0]
        noise_level=np.std(butter_bandpass_filter(filtered_data[i],0.8*fs/2.,fs/2.-1,fs))#get high frequency component
        pks=get_peak_indices(filtered_data,poration_splits=poration_splits,buffer=buffer,trace_idx=i) #buffer prevents selection around poration pulses
        #snip=data[i]
        #snip=snip/np.max(snip[start_t*fs:end_t*fs])
        peaks.append(pks)
        c_idx.append(i)
        noise_levels.append(noise_level)
    return peaks,c_idx,noise_levels
    
def get_good_peak_pairs(peaks,c_idx,noise_levels,data,npoints_beforepeak,npoints_afterpeak):
    def span(x):
        return np.max(x)-np.min(x)
    pairs_0,pairs_1,pairs_0_sp,pairs_0_np,pairs_0_sn,pairs_1_sp,pairs_1_np,pairs_1_sn,times=[],[],[],[],[],[],[],[],[]
    track_idx=[]
    ppeak=[]
    shift,mag=[],[]

        # added
    data_0_s=butter_bandpass_filter(data[c_idx[0]],lowcut_iap_sn,highcut_iap_sn,fs)
    data_0_n=butter_bandpass_filter(data[c_idx[0]],highcut_iap_sn+0.001,fs/2.-1,fs)
    data_1_s=butter_bandpass_filter(data[c_idx[1]],lowcut_iap_sn,highcut_iap_sn,fs)
    
    #added
    for idx,peak_0 in enumerate(peaks[0]):
        if (peak_0>npoints_beforepeak)&(peak_0<len(data.T)-npoints_beforepeak-1):
            data_0=data[c_idx[0]][peak_0-npoints_beforepeak:peak_0+npoints_afterpeak]
            data_1=data[c_idx[1]][peak_0-npoints_beforepeak:peak_0+npoints_afterpeak] 
            # added
            span_0=span(data_0)
            span_1=span(data_1)
            snr_0=span_0/noise_levels[0]
            snr_1=span_1/noise_levels[1]

            if (span_0 >= 500):
              if (snr_0>50)&(snr_1>50):
                  if ((np.argmax(data_0))>npoints_beforepeak-300)&(np.argmax(data_0)<npoints_beforepeak+300): #sometimes data accelerates unexplicably making weird shifted peaks
                    pt = np.argmax(data_0_s[peak_0-npoints_beforepeak:peak_0+npoints_afterpeak] )+peak_0-npoints_beforepeak
                    data_0=data[c_idx[0]][pt-npoints_beforepeak:pt+npoints_afterpeak]
                    data_1=data[c_idx[1]][pt-npoints_beforepeak:pt+npoints_afterpeak] 
                    s0=data_0_s[pt-npoints_beforepeak:pt+npoints_afterpeak]
                    n0=data_0_n[pt-npoints_beforepeak:pt+npoints_afterpeak]
                    s1=data_1_s[pt-npoints_beforepeak:pt+npoints_afterpeak] 
                    n1=data_1_n[pt-npoints_beforepeak:pt+npoints_afterpeak] 
                    sp_0,npower_0,sn_0 = singal_noise_power(s0,n0)
                    sp_1,npower_1,sn_1 = singal_noise_power(s1,n1)
            
            
                    pairs_0.append(data_0)
                    pairs_1.append(data_1)
                    times.append(pt * 0.0002)
                    pairs_0_sp.append(sp_0)
                    pairs_1_sp.append(sp_1)
                    pairs_0_np.append(npower_0)
                    pairs_1_np.append(npower_1)
                    pairs_0_sn.append(sn_0)
                    pairs_1_sn.append(sn_1)
    return pairs_0,pairs_1, pairs_0_sp, pairs_1_sp,pairs_0_np,pairs_1_np,pairs_0_sn,pairs_1_sn,times




def moving_filter(data, window_size):
  i = 0
  moving_averages = []
  while i < len(data) - window_size + 1:
      this_window = data[i : i + window_size]

      window_average = sum(this_window) / window_size
      moving_averages.append(window_average)
      i += 1

  for i in range(window_size - 1):
      moving_averages.append(data[len(data) - (window_size - i)])

  return moving_averages

# NEW GET_APDS RETURNS CORRECT 10 APDS
def get_apds(intra_trace, two_peak_cutoff):
  intra_trace = np.asarray(moving_filter(intra_trace, 20))
  period = pd.Series(intra_trace)
  std = period.rolling(window=20).std()
  stdofstd = np.std(std[19:])
  stdMedian = np.median(std[19:])

  start_loc = np.where(std == np.max(std[19:1500]))[0][0]

  while (start_loc >= 1 and std[start_loc] > stdMedian + 0.2 * stdofstd):
    start_loc = start_loc - 1;


  w1 = []
  h1 = []
  l1 = []

  period = period[start_loc:]
  two_peak_cutoff = two_peak_cutoff - start_loc 
  height = max(intra_trace) - min(intra_trace)
  height_cutoff = 0.85 * height

  locs = np.where(intra_trace[two_peak_cutoff:] > height_cutoff)[0] + two_peak_cutoff
  if len(locs) == 0:
  #for relheight in np.linspace(0.05,0.9,15):
    for relheight in np.linspace(0.1,1,10):
      w,h,l,r=peak_widths(period,find_peaks(period,distance=len(intra_trace) - 500)[0],rel_height=relheight)
      w1.append(w[0])
      h1.append(h[0])
      l1.append(l[0] + start_loc)
      
  else:
    distance = locs[0] - start_loc - 500
    for relheight in np.linspace(0.1,1,10):
      w,h,l,r=peak_widths(period,find_peaks(period[0:locs[0]-200], distance=distance)[0],rel_height=relheight)
      w1.append(w[0])
      h1.append(h[0])
      l1.append(l[0] + start_loc)
  return np.asarray(w1), np.asarray(h1), np.asarray(l1)

def get_all_apds(all_intras):
  APD_widths=[]
  for i in range(len(all_intras)):
    intra = all_intras[i]
    w1, h1, l1 = get_apds(intra, round(len(intra) * 0.7))
    APD_widths.append(w1)
  APD_widths = np.asarray(APD_widths)
  all_apds = APD_widths
  return all_apds

class Recording:
  def __init__(self, recording, intra_channel, extra_channel, window_size=4000,  buffer=50000):
    pair = [intra_channel, extra_channel]
#
    data,channel_to_elec,filtered_data,poration_splits=get_paired_dat_and_info(recording, pair)
# here it finds the peaks indicies
    peaks,c_idx,noise_levels=get_peaks_from_both_channels(pair,data,filtered_data,channel_to_elec,poration_splits,buffer)
# here it finds the peaks indicies  
    intras,extras,intras_sp,extras_sp,intras_np,extras_np,intras_sn,extras_sn,times= get_good_peak_pairs(peaks,c_idx,noise_levels,data,npoints_beforepeak=1000,npoints_afterpeak=window_size-1000)

    width_intras=[peak_widths(p,peaks=[np.argmax(p)],rel_height=0.5)[0][0] for p in intras]
    width_extras=[peak_widths(p,peaks=[np.argmax(p)],rel_height=0.5)[0][0] for p in extras]

    width_intras,width_extras = np.asarray(width_intras),np.asarray(width_extras)

    intras,extras,times=np.asarray(intras),np.asarray(extras),np.asarray(times)
    bool_cleanup = (width_intras>100)&(width_extras>0)&(width_extras<30) #make sure extras are not intra signals, and weird stuff (0 width) are cleaned

    # print(intras)
    # print(intras.shape)
    # print(np.array(intras_sp))
    # print(np.array(intras_sp).shape)
    # print(bool_cleanup)
    intras = intras[bool_cleanup]
    extras = extras[bool_cleanup]
    times = times[bool_cleanup]

    intras_sp = np.array(intras_sp)[bool_cleanup]
    extras_sp = np.array(extras_sp)[bool_cleanup]
    intras_np = np.array(intras_np)[bool_cleanup]
    extras_np = np.array(extras_np)[bool_cleanup]
    intras_sn = np.array(intras_sn)[bool_cleanup]
    extras_sn = np.array(extras_sn)[bool_cleanup]

    apds = get_all_apds(intras)

  # condition on eap
    b = extras - extras[:, [0]]
    condition1 = np.abs(b[:, -1]) < 100
  # condition 2
    condition2 = intras_sn>45
    indices = np.where(condition1 &condition2 )[0]
    indices = indices[2:-2]

    
    self.extras_raw = extras[indices];
    self.intras_raw = intras[indices];
    self.intras_sp = intras_sp[indices];
    self.extras_sp = extras_sp[indices];
    self.intras_np = intras_np[indices];
    self.extras_np = extras_np[indices];
    self.intras_sn = intras_sn[indices];
    self.extras_sn = extras_sn[indices];
    self.apds = apds[indices];
    self.times = times[indices];
    self.recording_info = [recording, intra_channel, extra_channel];

class MYDataset:
  
  def __init__(self, all_recordings, recs_to_include, intra_normalization=None, extra_normalization=None, test_size=0.25, val_size=.25, seedd = 42):
    self.intras_raw = []
    self.extras_raw = []
    self.intras_sp = []
    self.extras_sp = []
    self.intras_np = []
    self.extras_np = []
    self.intras_sn = []
    self.extras_sn = []
    self.apds = []
    self.times = []
    self.recordings = []
    self.recording_list = []

    self.intras_normalized = []
    self.extras_normalized = []

    for recording in recs_to_include:
      self.add_recording(all_recordings[recording])

    self.intras_raw = np.asarray(self.intras_raw)
    self.extras_raw = np.asarray(self.extras_raw)
    self.intras_sp = np.asarray(self.intras_sp)
    self.extras_sp = np.asarray(self.extras_sp)
    self.intras_np = np.asarray(self.intras_np)
    self.extras_np = np.asarray(self.extras_np)
    self.intras_sn = np.asarray(self.intras_sn)
    self.extras_sn = np.asarray(self.extras_sn)
    self.apds = np.asarray(self.apds)
    self.times = np.asarray(self.times)
    self.recordings = np.asarray(self.recordings)
    self.recording_list = np.asarray(self.recording_list)

    if intra_normalization == None:
      self.intras_normalized = self.intras_raw
    else: 
      for intra in self.intras_raw:
        self.intras_normalized.append(intra_normalization(intra))
      self.intras_normalized = np.asarray(self.intras_normalized)
    
    if extra_normalization == None:
      self.extras_normalized = self.extras_raw
    else: 
      for extra in self.extras_raw:
        self.extras_normalized.append(extra_normalization(extra))
      self.extras_normalized = np.asarray(self.extras_normalized)


    idx_traine,idx_test=train_test_split(np.arange(len(self.intras_raw)),test_size= test_size, random_state=seedd) 
    idx_train,idx_val=train_test_split(idx_traine, test_size=val_size, random_state=seedd)

    self.intras_training = self.intras_normalized[idx_train]
    self.intras_val = self.intras_normalized[idx_val]
    self.intras_testing = self.intras_normalized[idx_test]

    self.extras_training = self.extras_normalized[idx_train]
    self.extras_val = self.extras_normalized[idx_val]
    self.extras_testing = self.extras_normalized[idx_test]

    self.apds_training = self.apds[idx_train]
    self.apds_val = self.apds[idx_val]
    self.apds_testing = self.apds[idx_test]

    self.recordings_training = self.recordings[idx_train]
    self.recordings_val = self.recordings[idx_val]
    self.recordings_testing = self.recordings[idx_test]

    self.times_training = self.times[idx_train]
    self.times_val = self.times[idx_val]
    self.times_testing = self.times[idx_test]


  def add_recording(self, recording):
    for i in range(len(recording.intras_raw)):
        self.intras_raw.append(recording.intras_raw[i])
        self.extras_raw.append(recording.extras_raw[i])
        self.apds.append(recording.apds[i])
        self.times.append(recording.times[i])
        self.recordings.append(recording.recording_info)
        self.intras_sp.append(recording.intras_sp[i])
        self.extras_sp.append(recording.extras_sp[i]) 
        self.intras_np.append(recording.intras_np[i])
        self.extras_np.append(recording.extras_np[i])
        self.intras_sn.append(recording.intras_sn[i])
        self.extras_sn.append(recording.extras_sn[i])
    self.recording_list.append(recording.recording_info)





#Example creating Recording object

# Commented out IPython magic to ensure Python compatibility.


# os.chdir('drive/Shareddrives/NanoEngineering - BioNE Lab  /Ongoing Projects/NEA AI_Copy of Stanford Files/recording files/All Recording Files/4000 Point Spikes')

# recordings = sorted([el for el in os.listdir('./') if el.endswith('.h5')])
# recordings

#saving info from .h file in Recording object
# example_recording = Recording('2020AUGW1_D1_IPSCCM_DOD12_DIC49_0001.h5', 83, 64)

#saving Recording object


# with open('example_recording.pkl', 'wb') as outp:
  # pickle.dump(example_recording, outp, pickle.HIGHEST_PROTOCOL)

# Commented out IPython magic to ensure Python compatibility.
# %cd ../../..


# Commented out IPython magic to ensure Python compatibility.
# %cd 'Data'

#Example creating Dataset

#Loads dict of all Recordings objects

# os.chdir('/content/gdrive/Shared drives/NanoEngineering - BioNE Lab  /Ongoing Projects/NEA AI_Copy of Stanford Files/Data')

# with open('all_recordings.pkl', 'rb') as inp:
#   all_recordings = pickle.load(inp)

#Normalization functions (ones we used orginally)
def function1(intra):
  span=max(intra) - intra[0]
  return (intra-(intra)[0])/span

def function2(extra):
  std = np.std(extra[1150:1350])
  return ((extra-np.mean(extra))/(std * 60))

def double_spike_index_remover(dataset_4, d_4):
    dataset_4.extras_unseen = d_4.extras_normalized
    dataset_4.intras_unseen = d_4.intras_normalized
    double_spike_ind_train = []
    for i in range(len(dataset_4.extras_normalized)):
      y = dataset_4.extras_normalized[i]
      if (abs(y[2000:]).max()) > 0.17:
        double_spike_ind_train.append(i)
    lengths = len(dataset_4.extras_normalized)
    dataset_4.extras_normalized2 =dataset_4.extras_normalized[[i for i in range(lengths) if i not in double_spike_ind_train]]
    dataset_4.intras_normalized2 = dataset_4.intras_normalized[[i for i in range(lengths) if i not in double_spike_ind_train]]
    double_spike_ind_train = []
    for i in range(len(dataset_4.intras_training)):
      y = dataset_4.extras_training[i]
      if (abs(y[2000:]).max()) > 0.17:
        double_spike_ind_train.append(i)
    lengths = len(dataset_4.extras_training)
    dataset_4.extras_training2 = dataset_4.extras_training[[i for i in range(lengths) if i not in double_spike_ind_train]]
    dataset_4.intras_training2 = dataset_4.intras_training[[i for i in range(lengths) if i not in double_spike_ind_train]]
    double_spike_ind_val = []
    for i in range(len(dataset_4.intras_val)):
      y = dataset_4.extras_val[i]
      if (abs(y[2000:]).max()) > 0.17:
        double_spike_ind_val.append(i)
    lengths = len(dataset_4.extras_val)
    dataset_4.extras_val2 = dataset_4.extras_val[[i for i in range(lengths) if i not in double_spike_ind_val]]
    dataset_4.intras_val2 = dataset_4.intras_val[[i for i in range(lengths) if i not in double_spike_ind_val]]
    double_spike_ind_test = []
    for i in range(len(dataset_4.extras_unseen)):
      y = dataset_4.extras_unseen[i]
      if (abs(y[2000:]).max()) > 0.17:
        double_spike_ind_test.append(i)
    lengths = len(dataset_4.extras_unseen)
    dataset_4.extras_unseen2 = dataset_4.extras_unseen[[i for i in range(lengths) if i not in  double_spike_ind_test]]
    dataset_4.intras_unseen2 = dataset_4.intras_unseen[[i for i in range(lengths) if i not in  double_spike_ind_test]]



