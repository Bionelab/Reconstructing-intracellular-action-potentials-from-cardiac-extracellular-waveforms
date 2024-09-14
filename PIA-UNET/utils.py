# !pip install tsaug --upgrade --quiet a
# !pip install McsPyDataTools --upgrade --quiet 
# !python3 -m pip install --upgrade pip
# !pip install shap
# !pip install optuna
# !pip install xgboost
import os
import sys
import itertools
import pickle
import glob
import pickle
import random
import seaborn as sns
import numpy as np
import pandas as pd
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
# general_utils_dir = os. path.join(os.path.dirname(os.getcwd()), 'general_utils/')
# sys.path.append(general_utils_dir)
# from load_recording_data import *
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.signal import savgol_filter,find_peaks,peak_widths,argrelextrema
from sklearn.model_selection import KFold, train_test_split
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.utils import shuffle
import re
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from matplotlib.ticker import ScalarFormatter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from constants import *

def data_prep_test(data, test, raw=False, max_samples=400, limit_keys=None):
    """
    Prepares test data by extracting and optionally limiting the number of samples per index.

    Parameters:
    - data: Dictionary containing the dataset.
    - test: List of keys to extract the test data.
    - raw: Boolean indicating whether to use raw data or processed data.
    - max_samples: Maximum number of samples to include per index for keys in limit_keys.
    - limit_keys: List of keys for which the max_samples limit should be applied. If None, apply to all keys.
    
    Returns:
    - intras: Array of selected intracellular data samples.
    - extras: Array of selected extracellular data samples.
    """
    
    intras_test = []
    extras_test = []

    # Extract test data based on the 'raw' flag
    for key in test:
        if raw:
            intra_data = data[(key, 'intra_raw')]
            extra_data = data[(key, 'extra_raw')]
        else:
            intra_data = data[(key, 'intra')]
            extra_data = data[(key, 'extra')]

        # Get unique samples and their indices
        intra_unique, intra_indices = np.unique(intra_data, axis=0, return_index=True)
        extra_unique, extra_indices = np.unique(extra_data, axis=0, return_index=True)

        # Choose the smaller set of unique samples to align intra and extra
        if len(intra_unique) < len(extra_unique):
            chosen_ind = intra_indices
        else:
            chosen_ind = extra_indices

        # Apply max_samples only if the key is in limit_keys
        if limit_keys is None or key in limit_keys:
            # Limit the number of samples to max_samples if needed
            if len(chosen_ind) > max_samples:
                step_size = max(1, len(chosen_ind) // max_samples)
                chosen_ind = chosen_ind[::step_size][:max_samples]

        # Append limited data for this test index
        intras_test.append(intra_data[chosen_ind])
        extras_test.append(extra_data[chosen_ind])

    # Concatenate the limited test data
    intras = np.concatenate(intras_test)
    extras = np.concatenate(extras_test)
    
    return intras, extras



def function1_min_0(intra):
    min_val = min(intra[1000:])
    span = max(intra) - min_val
    return [(val - min_val) / span for val in intra]


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = 2 * (array - min_value) / (max_value - min_value) - 1
    return normalized_array

 
def name_from_dic  (dic, data_num):
    file_name = 'data-'+str(data_num)+'__'
    for name,val in dic.items():
        file_name = file_name + name+'-'+str(val)+'__'
    file_name = file_name+'.h5'
    return file_name

def name_to_dic  (name):
    dic_info = {}
    name_splitted = name.split('__')[2:-1]
    for i in name_splitted:
        key = i.split('-')[0]
        val = i.split('-')[1]
        dic_info[key]=val
    return dic_info

def get_file_names(loc):
    files = glob.glob(loc+'/*.h5')
    file_names = [os.path.basename(file_path) for file_path in files]
    return file_names


def find_longest_chunk_between_thresholds(data, start_threshold = 0.13, end_threshold = 0.1):
    """
    Finds the start and end indices of the longest consecutive chunk in 'data'
    where the values start above 'start_threshold' and go down to 'end_threshold'.

    Parameters:
    - data: numpy array or list of numerical values.
    - start_threshold: numerical threshold where the chunk should start.
    - end_threshold: numerical threshold where the chunk should end.

    Returns:
    - (start, end): tuple containing the start and end indices of the longest chunk.
                    Returns (None, None) if no such chunk is found.
    """
    data = np.array(data)
    
    # Step 1: Identify where the data first goes above the start threshold
    above_start_threshold = data > start_threshold
    
    consecutive_chunks = []
    start = None
    for i in range(len(above_start_threshold)):
        if above_start_threshold[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                # Check if this chunk ends when it goes below the end threshold
                if data[i] < end_threshold:
                    consecutive_chunks.append((start, i-1))
                    start = None

    # Add the last chunk if it ends at the end of the array
    if start is not None and data[-1] >= end_threshold:
        consecutive_chunks.append((start, len(above_start_threshold) - 1))

    # Find the longest chunk
    longest_chunk = max(consecutive_chunks, key=lambda x: x[1] - x[0]) if consecutive_chunks else (None, None)

    return longest_chunk

def calculate_mae_between_chunks(array1, array2, start_threshold= 0.13, end_threshold= 0.1):
    """
    Calculates the Mean Absolute Error (MAE) between corresponding samples in array1 and array2,
    considering only the values within the longest chunk above the threshold.

    Parameters:
    - array1: numpy array of shape (n_samples, n_features)
    - array2: numpy array of shape (n_samples, n_features)
    - threshold: numerical threshold value to define the chunk.

    Returns:
    - mae: Mean Absolute Error for the longest chunk across all samples.
    """
    assert array1.shape == array2.shape, "Both arrays must have the same shape."
    
    maes = []
    chunk1s=[]
    chunk2s=[]
    
    for i in range(array1.shape[0]):
        sample1 = array1[i]
        sample2 = array2[i]
        
        # Find the longest chunk for the current sample
        start, end = find_longest_chunk_between_thresholds(sample1, start_threshold, end_threshold)
        
        if start is not None and end is not None:
            # Calculate MAE for the chunk
            chunk1 = sample1[start:end+1]
            chunk2 = sample2[start:end+1]
            mae = np.mean(np.abs(chunk1 - chunk2))
            maes.append(mae)
            chunk1s.append(chunk1)
            chunk2s.append(chunk2)
        else:
            # If no valid chunk is found, append NaN or 0 as per your requirement
            maes.append(np.nan)  # or append(0) if preferred
    
    # Return the MAE across all samples (ignoring NaN values if present)
    return np.array(maes),chunk1s,chunk2s



def moving_filter(arr, window_size):
    # Simple moving average filter
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def get_apds_single_trace(intra_trace, two_peak_cutoff):
    intra_trace = np.asarray(moving_filter(intra_trace, 20))
    period = pd.Series(intra_trace)
    std = period.rolling(window=20).std()
    stdofstd = np.std(std[19:])
    stdMedian = np.median(std[19:])

    start_loc = np.argmax(std[19:1500]) + 19

    while start_loc >= 1 and std[start_loc] > stdMedian + 0.2 * stdofstd:
        start_loc -= 1

    w1, h1, l1 = [], [], []
    period = period[start_loc:]
    two_peak_cutoff = max(0, two_peak_cutoff - start_loc)

    locs = np.where(intra_trace[two_peak_cutoff:] > 0.85)[0] + two_peak_cutoff
    if len(locs) == 0:
        for relheight in np.linspace(0.1, 1, 10):
            w, h, l, _ = peak_widths(period, find_peaks(period, distance=3500)[0], rel_height=relheight)
            w1.append(w[0])
            h1.append(h[0])
            l1.append(l[0] + start_loc)
    else:
        distance = max(1, locs[0] - start_loc - 500)
        for relheight in np.linspace(0.1, 1, 10):
            w, h, l, _ = peak_widths(period[:locs[0] - 200], find_peaks(period[:locs[0] - 200], distance=distance)[0], rel_height=relheight)
            w1.append(w[0])
            h1.append(h[0])
            l1.append(l[0] + start_loc)

    return np.asarray(w1), np.asarray(h1), np.asarray(l1)

def get_all_apds_multiprocessing(all_intras):
    # Using multiprocessing for parallel computation
    with ProcessPoolExecutor() as executor:
        apd_results = list(executor.map(partial(get_apds_single_trace, two_peak_cutoff=round(len(all_intras[0]) * 0.7)), all_intras))
    APD_widths = np.array([w for w, h, l in apd_results])
    return APD_widths

def return_apds_quantile(df, dic_eaps):
    preds1 = np.array([dic_eaps[(i, 'p1_smooth')] for i in df.index])
    preds2 = np.array([dic_eaps[(i, 'p2_smooth')] for i in df.index])
    preds3 = np.array([dic_eaps[(i, 'p3_smooth')] for i in df.index])

    # Using multiprocessing to process all traces in parallel
    apd1 = get_all_apds_multiprocessing(preds1.reshape(-1, 8000))
    print('ap1 done')
    apd2 = get_all_apds_multiprocessing(preds2.reshape(-1, 8000))
    print('ap2 done')
    apd3 = get_all_apds_multiprocessing(preds3.reshape(-1, 8000))
    print('ap3 done')

    apd_col1 = ['APD_P1_' + str(i) for i in range(1, 11)]
    apd_col2 = ['APD_P2_' + str(i) for i in range(1, 11)]
    apd_col3 = ['APD_P3_' + str(i) for i in range(1, 11)]

    df2 = df.copy()
    df2.loc[df2.index, apd_col1] = apd1
    df2.loc[df2.index, apd_col2] = apd2
    df2.loc[df2.index, apd_col3] = apd3

    # Filter wrong predictions of APD
    for cols in ['APD_P1_9', 'APD_P3_7', 'APD_P1_5']:
        df2 = df2[df2[cols] > 2000]
        
    return apd1, apd2, apd3, df2

## moving singals to match their location, inc case of slight difference, its because to measure correlation correclety after
def autro_correct(preds, intras_test_):
    preds_new = []
    intras_new = []
    target_length = 8000  # The desired length after padding

    for i, j in zip(preds, intras_test_):

        # Compute correlation and find the best lag
        correlation = np.correlate(i - np.mean(i), j - np.mean(j), mode='full')
        lag = np.argmax(correlation) - (len(j) - 1)
    
        # Shift the arrays based on the best lag
        if lag > 0:
            i_shifted = i[lag:]
            j_shifted = j[:len(i_shifted)]
        else:
            j_shifted = j[-lag:]
            i_shifted = i[:len(j_shifted)]

        # Pad the arrays to reach the target length of 8000
        i_padded = np.pad(i_shifted, (0, max(0, target_length - len(i_shifted))), 'constant')
        j_padded = np.pad(j_shifted, (0, max(0, target_length - len(j_shifted))), 'constant')

        # Append the padded arrays to the new lists
        preds_new.append(i_padded)
        intras_new.append(j_padded)

    return np.array(preds_new), np.array(intras_new)
