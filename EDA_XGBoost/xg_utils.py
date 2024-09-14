import os
import sys
import seaborn as sns

from colorsys import rgb_to_hls, hls_to_rgb

import matplotlib.gridspec as gridspec
from scipy import interpolate
general_utils_dir = os. path.join(os.path.dirname(os.getcwd()), 'general/')
sys.path.append(general_utils_dir)
# from general_utils import *
# from load_recording_data import *
from peak_finder import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import glob
import pickle
import random
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.signal import savgol_filter,find_peaks,peak_widths,argrelextrema
from sklearn.model_selection import KFold, train_test_split
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.utils import shuffle
import re
import jenkspy
from scipy.interpolate import interp1d
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
import shap
from matplotlib.ticker import ScalarFormatter
from concurrent.futures import ProcessPoolExecutor
from functools import partial


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



def detect_changes_iap(signal, cl):
    breaks = jenkspy.jenks_breaks(signal, n_classes=cl)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[0],change_points[-1]
def detect_changes_iap2(signal):
    np.random.seed(0)
    random.seed(0)
    algo = rpt.Binseg(model='rbf', jump=20).fit(signal)
    result = algo.predict(n_bkps=2)
    change_points = [i for i in result if i < len(signal)]
    return change_points
    
def iap_ch(signal):
    window_length, poly_order = 130, 5
    sdata = savgol_filter(signal, window_length, poly_order)
    sdata = savgol_filter(sdata, window_length, poly_order)
    x1 = np.argmax(sdata)
    window_length, poly_order = 50, 3
    sdata = savgol_filter(signal[x1-300:x1-20], window_length, poly_order)
    sdata_g = np.gradient(sdata)
    x0_ = detect_changes_iap(sdata_g, 2)[0] + x1-300 
    # x0_ = detect_changes_iap2(sdata_g)[0] + x1-200
    y0_ = signal[x0_]
    window_length, poly_order = 1000, 5
    sdata = savgol_filter(signal[x1+200:], window_length, poly_order)
    sdata_g = np.gradient(sdata)
    indexes3 = detect_changes_iap2(sdata_g)
    x1_ = indexes3[0]
    x2_ = indexes3[1]
    x1_ = x1_ + x1 + 200
    x2_ = x2_ + x1 + 200
    y1_,y2_ = signal[x1_],signal[x2_]
    return x0_,x1_, x2_,y0_,y1_,y2_, x1
    

def detect_changes_eap1(signal):#bp1
    breaks = jenkspy.jenks_breaks(signal, n_classes=3)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[-1]


def detect_changes_eap2(signal):#bp2
    breaks = jenkspy.jenks_breaks(signal, n_classes=2)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[0]


def detect_changes_eap3(signal):#bp3
    breaks = jenkspy.jenks_breaks(signal, n_classes=3)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[1]

def eap_x3(i, x1, sdata):
    initial_gap = x1+ 300
    i2= i[initial_gap:] # after spike
    sdata2 = sdata[initial_gap:]
    index_above_threshold = np.argmax(i2 >  0.7 * i[x1])
    # print('index_above_threshold',index_above_threshold)
    i3 = i2[:index_above_threshold-200] # # before next spike
    sdata3 = sdata2[:index_above_threshold-200]
    window_length = 500
    poly_order = 3

    while window_length > poly_order:
        try:
            # Apply the Savitzky-Golay filter twice
            # sdata = savgol_filter(sdata.reshape(-1), window_length, poly_order)
            # Find the minimum point after smoothing
            x3 = np.argmin(sdata3) + initial_gap
            y3 = np.mean(i[int(x3)-10:int(x3)+10])
            # Return immediately if no error occurs
            return x3, y3
        except ValueError as e:
            # Reduce the window length by 3/4 if an error occurs
            window_length = int(window_length * 3 / 4)
            print(f"Reducing window length to {window_length} due to error: {e}")
    # If all attempts fail, raise an error
    raise ValueError("Could not apply Savitzky-Golay filter with the given parameters.")
 
def raise_extra(data,x3 ):
    window_length, poly_order = min(200,len(data)-1), min(5,len(data)-1)
    i2 = savgol_filter(data, window_length, poly_order)
    try:
        z = i2[x3:x3+1000]
    except:
        z = i2[x3:]
    rate= (z[-1]-z[0])/len(z)
    return rate

def snr(i):
    signal = i[-3000:]
    window_length = len(signal)
    poly_order = 3
    smoothed_signal = savgol_filter(signal, window_length, poly_order)
    noise = np.std(signal-smoothed_signal)
    return np.max(i)/noise
    
def eap_ch1(i,x1):
    window_length, poly_order = 75,5
    i_n2 = i[0:x1].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order)
    sdata2 = savgol_filter(sdata2, window_length, poly_order)
    
    sdata_g2 = np.gradient(sdata2)
    sdata_g2 = savgol_filter(sdata_g2, window_length, poly_order)
    ch_p2 = detect_changes_eap1(sdata_g2)
    x_ch2 = ch_p2
    y_ch2 = sdata2[ch_p2]
    return x_ch2, y_ch2
    
def eap_ch2_2(i):
    x1 = np.argmax(i)
    window_length, poly_order = 75,5
    x2 = np.argmin(i[x1:x1+100])+x1
    i_n2 = i[x2:x2+150].copy()
    i_n3 = i[x1:x2].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order)
    sdata2 = savgol_filter(sdata2, window_length, poly_order)
    
    sdata_g2 = np.gradient(sdata2)
    sdata_g2 = savgol_filter(sdata_g2, window_length, poly_order)
    ch_p2 = detect_changes_eap2(sdata_g2[10:-10])+10
    x_ch2 = ch_p2+x2
    y_ch2 = sdata2[ch_p2]
    return x_ch2, y_ch2
    
def eap_ch2_3(i,x2):
    window_length, poly_order = 75,5
    i_n2 = i[x2:x2+150].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order) 
    sdata_g2 = np.gradient(sdata2)
    sdata_g2 = savgol_filter(sdata_g2, window_length, poly_order)
    ch_p2 = detect_changes_eap2(sdata_g2[10:-10])+10
    x_ch2 = ch_p2+x2
    y_ch2 = sdata2[ch_p2]
    return x_ch2, y_ch2

def eap_ch2_4(i,x2):
    window_length, poly_order = 75,5
    i_n2 = i[x2:x2+400].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order)
    sdata2 = savgol_filter(sdata2, window_length, poly_order)
    index1 =  np.arange(x2,x2+400)
    sdata2_inv = sdata2[::-1]
    sindex_inv = index1[::-1]
    sdata_g2 = np.gradient(sdata2_inv)
    sdata_g2 = savgol_filter(sdata_g2, window_length, poly_order)
    ch_p2 = detect_changes_eap2(sdata_g2[10:-10])+10
    ch_p2 = (400-ch_p2)+50
    y_ch2 = sdata2[ch_p2]
    x_ch2 = ch_p2+x2
    return x_ch2, y_ch2


def eap_ch1_y_eq (ri,xch1,ych1,x1,x2):
    x = np.arange(x1,x2)
    f = interpolate.interp1d(x, ri[x1:x2])
    x_new = np.linspace(x[0], x[-1] , 5000) # Extrapolate 1000 points between x[0] and x2
    y_new = f(x_new)
    idx = np.argmin(np.abs(y_new - ych1))  # Find the index of the closest y value to y0
    x_value = x_new[idx]
    return x_value 

def eap_ch2_y_eq (ri,xch2,ych2,x1,x2):
    x = np.arange(x1,x2)
    f = interpolate.interp1d(x, ri[x1:x2])
    x_new = np.linspace(x[0], x[-1] , 5000) # Extrapolate 1000 points between x[0] and x2
    y_new = f(x_new)
    idx = np.argmin(np.abs(y_new - ych2))  # Find the index of the closest y value to y0
    x_value = x_new[idx]
    return x_value

def eap_df_generator_EDA (data, data_names ,distorted_dic, to_plot= False ):
    order_sn = 3
    window_length = 500
    poly_order = 3

    
    exp_ind_dic = {key: np.arange(len(data[key, 'extra'])) for key in data_names}
    dfs = []
    for item in exp_ind_dic.items():
        key, indicies = item
        for ind in indicies:
            if ind not in distorted_dic[key]:
                print(key, ind)
                i = data[key,'extra_norm1'][ind]  # normalized data
                ri = data[key,'extra_raw'][ind]   # raw data

                if 'mea' in key:
                    fs = 10000
                    lowcut_iap_sn = 0.01
                    highcut_iap_sn  = fs/32-1
                else:
                    fs = 5000
                    lowcut_iap_sn = 0.01
                    highcut_iap_sn  = fs/32-1
               

                sdata= butter_bandpass_filter(ri.reshape(1,-1), lowcut_iap_sn, highcut_iap_sn, fs, order=3)
                sdata2 = savgol_filter(sdata.reshape(-1), window_length, poly_order)

                x_ = np.argmax(ri[:1400])  # spike x
                i = i - np.mean(i[0:100])  # to make starting region around 0
                ri = ri - np.mean(ri[0:100])  # to make starting region around 0
                x1s, a_ = find_peaks(i[:1400], height=np.max(i[:1400])*0.6, distance=20)
                x1 = x1s[-1]  # first spike x
                y1, ry1 = i[x1], ri[x1]  # first spike y in normalized and raw
                x2s, a_ = find_peaks(-i[x1:x1+150], height=-np.min(i[x1:x1+150])*0.6, distance=20)
                # print(x2s)
                try:
                    x2 = x2s[-1] + x1
                except:
                    x2 = np.argmin(i[x1:x1+150]) + 1
                y2 = i[x2]
                ry2 = ri[x2]
                if True:
                    x3, ry3 = eap_x3(ri, x1,sdata2)  # i , chp+3
                    y3 = np.mean(i[x3-10:x3+10])
                    x_chp2, ry_chp2 = eap_ch2_3(ri, x2)
                    y_chp2 = i[x_chp2]
                    x_chp1, ry_chp1 = eap_ch1(ri, x1)
                    y_chp1 = i[x_chp1]
                    h1_n = y1 - y_chp1
                    h1 = ry1 - ry_chp1
                    x_chp1_eq = eap_ch1_y_eq(ri, x_chp1, ry_chp1, x1, x2)
                    x_chp2_eq = eap_ch2_y_eq(ri, x_chp2, ry_chp2, x1, x2)
                    rdecay = [ri[x_chp2+100] - ry3] / (x3 - x_chp2 - 100)
                    decay = [i[x_chp2+100] - y3] / (x3 - x_chp2 - 100)
                    h2 = ry_chp2 - ry2
                    h2_n = y_chp2 - y2
                    w3 = x3 - x_chp2
                    w1_2_3 = x3 - x_chp1
            
                    # Two options for w
                    w1 = x_chp1_eq - x_chp1
                    w2 = x_chp2 - x_chp2_eq
                    w1_2 = x_chp2 - x_chp1
            
                    w1_p = x1 - x_chp1
                    w2_p = x_chp2 - x2
                    w1_2_p = x2 - x_chp1
            
                    h3 = ry_chp2 - ry3
                    h3_n = y_chp2 - y3
            
                    h1_h2 = h1 / h2
                    h3_h1 = h3 / h1
                    h3_h2 = h3 / h2
                    s_n = snr(i)
                    rs_n = snr(ri)

                    if to_plot:
                    # Create side-by-side plots
                        fig, axes = plt.subplots(1, 2, figsize=(50, 10), )
                
                        # First plot with annotations and green points
                        axes[0].plot(ri)
                        
                        axes[0].plot([x1, x2, x_chp1, x_chp2, x3], [ri[x1], ri[x2], ri[x_chp1], ri[x_chp2], ri[x3]], 'ro', markersize=10)
                        axes[0].plot([x_chp1_eq, x_chp2_eq], [ri[x_chp1], ri[x_chp2]], 'ro', markersize=10)
                
                        # Annotate the red points
                        axes[0].text(x1, ri[x1], 'x1', fontsize=15, ha='right')
                        axes[0].text(x2, ri[x2], 'x2', fontsize=15, ha='right')
                        axes[0].text(x_chp1, ri[x_chp1], 'x_chp1', fontsize=15, ha='right')
                        axes[0].text(x_chp2, ri[x_chp2], 'x_chp2', fontsize=15, ha='right')
                        axes[0].text(x3, ri[x3], 'x3', fontsize=15, ha='right')
                
                        # Annotate the green points
                        axes[0].text(x_chp1_eq, ri[x_chp1], 'x_chp1_eq', fontsize=15, ha='left', color='red')
                        axes[0].text(x_chp2_eq, ri[x_chp2], 'x_chp2_eq', fontsize=15, ha='left', color='red')
                
                        axes[0].set_xlim(600, 1100)
                        # axes[0].tick_params(axis='both', which='major', labelsize=15)
                        # axes[0].set_title('Plot 1: Annotated Peaks', fontsize=20)
                
                        # Second plot showing (x1, y1), (x2, y2), and (x3, y3)
                        axes[1].plot(ri)
                        axes[1].plot(sdata.reshape(-1), color = 'red')
                        axes[1].plot(sdata2, color = 'green')
                        axes[1].plot([x1, x2, x3,x_chp1,x_chp2], [ri[x1], ri[x2], ri[x3],ri[x_chp1], ri[x_chp2]], 'ro', markersize=20)  # Blue points for x1, x2, x3
                
                        # Annotate the blue points
                        axes[1].text(x1, ri[x1], f'(x1, y1)', fontsize=15, ha='right', color='red')
                        axes[1].text(x2, ri[x2], f'(x2, y2)', fontsize=15, ha='right', color='red')
                        axes[1].text(x3, ri[x3], f'(x3, y3)', fontsize=15, ha='right', color='red')
                
                        # axes[1].set_xlim(600, 1100)
                        # axes[1].tick_params(axis='both', which='major', labelsize=35)
                        # axes[1].set_title('Plot 2: (x1, y1), (x2, y2), (x3, y3)', fontsize=30)
                
                        # plt.tight_layout()
                        plt.show()
            
                    df = pd.DataFrame([{
                        't1': w1,
                        't2': w2,
                        'ts': w1_2,
                        't1_alt': w1_p,
                        't2_al': w2_p,
                        'ts_alt': w1_2_p,
                        't1+t2+td': w1_2_3,
                        'td': w3,
                        'v1': h1,
                        'v2': h2,
                        'v1_normalized': h1_n,
                        'v2_normalized': h2_n,
                        'v1/v2': h1_h2,
                        'vd/v1': h3_h1,
                        'vd/v2': h3_h2,
                        'vd': h3,
                        'vd_normalized': h3_n,
                        'x_chp2': x_chp2,
                        'y_chp2': ry_chp2,
                        'x_max': x1,
                        'y_max': y1,
                        'y_u': ry1,
                        'x_min': x2,
                        'Ny_min': y2,
                        'y_b': ry2,
                        'x_min2': x3,
                        'Ny_min2': y3,
                        'y_m': ry3,
                        'name': key,
                        'ind': ind,
                        'IR': raise_extra(i, x3),
                        'IR_raw':raise_extra(ri, x3),
                        'DR': decay[0],
                        'DR_raw':rdecay[0],
                        's/n': s_n,
                        's/n_raw': rs_n
                    }])
                    dfs.append(df)

        
    df_eap = pd.concat(dfs)
    
    eap_time_dep_cols  = ['t1', 't2','ts','td','t1_alt', 't2_al', 'ts_alt','t1+t2+td',]
    eap_timeamp_dep_slope_col = ['IR', 'DR']
    eap_amp_cols = [  'v1','v2','v1_normalized','v2_normalized','vd', 'vd_normalized']

   
    df_eap[eap_time_dep_cols] = df_eap[eap_time_dep_cols]/5000
    df_eap[eap_timeamp_dep_slope_col] = df_eap[eap_timeamp_dep_slope_col]*0.005
    df_eap[eap_amp_cols] = df_eap[eap_amp_cols]/1000
    return df_eap

def iap_df_generator_EDA(data, data_names ,distorted_dic, to_plot = False ):
    apd_cos = ['APD'+str(10*i) for i in range(1,11) ]
    exp_ind_dic = {key: np.arange(len(data[key, 'extra'])) for key in data_names}
    dfs = []
    for item in exp_ind_dic.items():
        key, indicies = item
        inds = [i for i in indicies if i not in distorted_dic[key] ]
        intras = [data[key,'intra'][i] for i in indicies if i not in distorted_dic[key] ] 
        sns = [data[key,'s/n'][i] for i in indicies if i not in distorted_dic[key] ] 
        if len (intras)>0:
            apds = get_all_apds(intras)
            if to_plot:
                for i, j in zip(intras, inds):
                    print(key, j)
                    w1_test, h1_test, l1_test = get_apds(i, 2800)
                    plt.figure(figsize=(10, 6))
                    plt.plot(i)
                    for w, h, l in zip(w1_test, h1_test, l1_test):
                        plt.plot([l, l + w], [h, h], marker='o')  # Draw a horizontal line from (l, h) to (l + w, h)
                    plt.show()
            print(apds.shape)
            df = pd.DataFrame()
            df[apd_cos]=apds
            df['iap_sn'] = sns
            df['name']=key
            df['ind'] = inds
            dfs.append(df)

    df_iap = pd.concat(dfs)
    df_iap[apd_cos] = df_iap[apd_cos]/5000
    return df_iap

def adjust_lightness(color, amount=5.0):
    """ Adjust the lightness of a given color """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    r, g, b = mcolors.to_rgb(c)
    h, l, s = rgb_to_hls(r, g, b)
    l = max(min(l * amount, 1.0), 0.0)
    return hls_to_rgb(h, l, s)



def feature_generater(data,all_keys,distorted_dic_xg,to_plot= False):
    df_eap_xg= eap_df_generator_EDA(data, all_keys ,distorted_dic_xg,  to_plot= to_plot)
    df_iap_xg= iap_df_generator_EDA(data, all_keys ,distorted_dic_xg )
    df_xg = pd.concat([df_eap_xg.set_index(['name','ind']),df_iap_xg.set_index(['name','ind'])], axis = 1)
    df_xg = df_xg[(df_xg['v1']>0) & (df_xg['v2']>0) & (df_xg['vd']>0)]
    return df_xg

def post_screening(df_xg,window_size=10, threshold_factor=3, to_exclude =['exp13_propranolol_W5_D1_i23_e13.pkl',                                                                       'exp00_dofetilide_W8_D2_i57_e47.pkl'] ):
    """
    Function to perform outlier detection and filtering based on rolling median and MAD.
    for post_screening of samples which their bp3 might be mistakenly computed
    
    Parameters:
    - df_xg: DataFrame containing the full dataset.
    - window_size: Size of the rolling window for median calculation.
    - threshold_factor: Factor for multiplying the Median Absolute Deviation (MAD) to set the outlier threshold.
    
    Returns:
    - datas: List of DataFrames with filtered data.
    """
    datas = []
    df_xg2 = df_xg.reset_index()
    
    for name in df_xg2['name'].unique():
        print(name)
        
        # Filter the data for the current name and set index to 'ind'
        full_data = df_xg2[df_xg2['name'] == name].set_index('ind').iloc[:]
        data__ = df_xg2[df_xg2['name'] == name].reset_index().set_index('ind')['td'].iloc[:]
        
        # Calculate the rolling median to capture the trend
        rolling_median = data__.rolling(window=window_size, center=True).median()
        
        # Calculate the deviation from the rolling median
        deviation = np.abs(data__ - rolling_median)
        
        # Calculate the Median Absolute Deviation (MAD)
        mad = deviation.median()
        
        # Define a threshold for outlier detection
        threshold = threshold_factor * mad
        
        # Create a boolean mask for filtering data
        mask = deviation < threshold
        
        # Filter the original data using the mask
        filtered_data = data__[mask]
        filtered_full_data = full_data[mask]
        
        # Get the indices of the filtered data
        inds_filtered = np.arange(len(data__))[mask]
        
        # Plot the original data, rolling median, and filtered data
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(data__)), data__, label='Original Data', alpha=0.4)
        plt.plot(np.arange(len(data__)), rolling_median, color='green', label='Rolling Median', linewidth=2)
        plt.scatter(inds_filtered, filtered_data, color='red', label='Filtered Data', alpha=0.8)
        plt.title(f"Outlier Detection for {name}")
        plt.xlabel('Index')
        plt.ylabel('w3')
        plt.legend()
        plt.show()
        
        # Store full or filtered data based on specific names
        if name in []:
            datas.append(full_data)
        else:
            datas.append(filtered_full_data)
    
    return pd.concat(datas)

