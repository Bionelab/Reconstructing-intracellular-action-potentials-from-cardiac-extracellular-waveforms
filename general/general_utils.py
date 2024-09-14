import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import find_peaks,peak_widths,argrelextrema,savgol_filter,butter,filtfilt
import random
import re
import math
import json
import h5py

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
  two_peak_cutoff = two_peak_cutoff - start_loc #try +

  locs = np.where(intra_trace[two_peak_cutoff:] > 0.85)[0] + two_peak_cutoff
  if len(locs) == 0:
  #for relheight in np.linspace(0.05,0.9,15):
    for relheight in np.linspace(0.1,1,10):
      w,h,l,r=peak_widths(period,find_peaks(period,distance=3500)[0],rel_height=relheight)
      w1.append(w[0])
      h1.append(h[0])
      l1.append(l[0] + start_loc)
      
  else:
    distance = locs[0] - start_loc - 500
    if distance < 1:
        print(distance)
        distance = 1
    for relheight in np.linspace(0.1,1,10):
      w,h,l,r=peak_widths(period,find_peaks(period[0:locs[0]-200], distance=distance)[0],rel_height=relheight)
      w1.append(w[0])
      h1.append(h[0])
      l1.append(l[0] + start_loc)
  return np.asarray(w1), np.asarray(h1), np.asarray(l1)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def filtered_data (data, fs = 5000):
    return butter_bandpass_filter(data,1,fs/2.-1,fs)



def normalize_array(intra):
    min_val = np.min(intra[:])
    span = np.max(intra) - min_val
    return (intra - min_val) / span 


def get_apd(spike):
    w1_test, h1_test, l1_test = get_apds(spike, 7000)
    pred_widths = np.asarray(w1_test)
    pred_ls= np.asarray(l1_test)
    return pred_widths, pred_ls




def average_correlation(array1, array2):
    correlation_matrix = np.corrcoef(array1, array2)
    average_correlation = correlation_matrix[0, 1]
    return average_correlation
