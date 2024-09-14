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
from peak_finder import *
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
import shap
from matplotlib.ticker import ScalarFormatter
import math
# general_utils_dir = os. path.join(os.path.dirname(os.getcwd()), 'general_utils/')
# sys.path.append(general_utils_dir)
from general_utils import *
from load_recording_data import *
from peak_finder import *
# import tsaug


def singal_noise_power(spike, noise):
    sp_ = np.mean(spike ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(noise ** 2)  # Square of the signal's amplitude
    sn_ = 20 * math.log10(sp_ / npower_)
    return sp_,npower_,sn_


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = 2 * (array - min_value) / (max_value - min_value) - 1
    return normalized_array

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


#bp1
def detect_changes_eap1(signal):
    breaks = jenkspy.jenks_breaks(signal, n_classes=3)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[-1]

#bp2
def detect_changes_eap2(signal):
    breaks = jenkspy.jenks_breaks(signal, n_classes=2)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[0]

#bp3
def detect_changes_eap3(signal):
    breaks = jenkspy.jenks_breaks(signal, n_classes=3)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[1]

def eap_x3(i, x_ch2):
    window_length = 500
    poly_order = 3
    while window_length > poly_order:
        try:
            # Apply the Savitzky-Golay filter twice
            sdata = savgol_filter(i[x_ch2 + 300:], window_length, poly_order)
            sdata = savgol_filter(sdata, window_length, poly_order)
            # Find the minimum point after smoothing
            x3 = np.argmin(sdata) + x_ch2 + 300
            y3 = np.min(sdata)
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
# no more x_ch2 50 % data
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
#return in correct units, also skipping 50% values for eap spike regions

def eap_ch2_3(i,x2):
    window_length, poly_order = 75,5
    i_n2 = i[x2:x2+150].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order)
    sdata2 = savgol_filter(sdata2, window_length, poly_order)    
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


def eap_df_generator_XG(dictionary_of_input):
    dict_items = list(dictionary_of_input.items())
    dfs = []
    for item in dict_items:
            key, value = item
    
            print(key)
            dataset_8 = train_data_sh(0.2, seed=7, UNSEEN8=[key])
            d_8 = MYDataset(all_recordings, [key], function1, function2, test_size=0.0001, val_size=0.001, seedd=seed)
            print(len(dataset_8.extras_unseen2), len(d_8.extras_raw))
            # print(dataset_8.extras_unseen_sn)
            for ind in value:
                # print(key, ind)
                i = dataset_8.extras_unseen2[ind]
                t_ = d_8.times[ind]
                ri = d_8.extras_raw[ind]
                x_ = np.argmax(ri)
                i = i - np.mean(i[0:100])
                ri = ri-np.mean(ri[0:100])

                x1s, a_  = find_peaks(i,height = np.max(i)*0.6, distance=20)
                x1_2 = x1s[-1]
                x1_1 = x1s[0]
                y1_1, y1_1, ry1_1 = i[x1_1], i[x1_1], ri[x1_1]
                y1_2, y1_2,ry1_2 = i[x1_2], i[x1_2], ri[x1_2]
                x2s, a_  = find_peaks(-i[x1_2:x1_2+150],height = -np.min(i[x1_1:x1_2+150])*0.6, distance=20)
                # try:
                #     x2= x2s[-1] + x1_2
                # except:
                x2 =  np.argmin(i[x1_2:x1_2+60])+x1_2
                # print(x2)
                y2,ry2 = i[x2], ri[x2]
  
                x3,ry3 = eap_x3(ri,x1_2+400) #i , chp+3
                _,y3 = eap_x3(i,x1_2+400) #i , chp+3
             
                x_chp1,y_chp1 = eap_ch1(i,x1_1)
                _,ry_chp1 = eap_ch1(ri,x1_1)
                
                x_chp2,y_chp2 = eap_ch2_3(i,x2)
                _,ry_chp2 = eap_ch2_3(ri,x2)
                # ry_chp2= ri[x_chp2]
                # ry_chp1= ri[x_chp1]


                h1 = ry1_1 - ri[:x1_1-200].mean()
                h1  = ry1_1  - ry_chp1
                decay = [ri[x_chp2+100]-ry3]/(x3-x_chp2-100)
                decay  =  [ry_chp2-ry3]/(x3-x_chp2-100)
                h2 = y_chp2 - ry2
                w3 = x3 - x_chp2
                w1_2_3 = x3 -x_chp1
                w1_2_3_s = x3 -x_chp1
                w1_2= x_chp2-x_chp1
                h3 = ry_chp2 - ry3
                
                h1_h2 = h1/h2
                h3_h1 = h3/h1
                h3_h2 = h3/h2
                s_n = snr(i)
                rs_n = snr(ri)
                
                sn_power =dataset_8.extras_unseen_sn[ind] 
#                 if h1 < 0 or h2 < 0 or h3<0:
#                     print('a',key, ind)
#                     plt.plot(ri)
#                     plt.plot([x1_1,x1_2,x2,x_chp1,x_chp2],[ry1_1,ry1_2,ry2,ry_chp1,ry_chp2], 'ro')
#                     plt.xlim(600,1100)
#                     plt.show()
#                     plt.plot(ri)

#                     # plt.plot([x2,],[ry2,], 'ro')
#                     # plt.plot([x_chp1_eq,x_chp2_eq],[ri[x_chp1],ri[x_chp2]], 'go')

#                     plt.plot([x1_1,x1_2,x2,x_chp1,x_chp2, x3],[ry1_1,ry1_2,ry2,ry_chp1,ry_chp2, ry3], 'ro')
#                     plt.show()

                df = pd.DataFrame([{
                    'sn_power':sn_power,
                       'x_chp1':x_chp1,
                    't':t_,
                    'y_chp1':ry_chp1,
                    
                  'w1+w2':w1_2,
        
                  'w3+w1+w2':w1_2_3,
                  'w3':w3,
                  'h1':h1,
                  'h2':h2,
                  'h1/h2':h1_h2,
                  'h3/h1':h3_h1,
                  'h3/h2':h3_h2,
                  'h3':h3,
                  'x_chp2':x_chp2,
                  'y_chp2':ry_chp2,
                  'x1': x1_1,
                  'y1': y1_1,
                  'ry1': ry1_1,
                  'x2': x2,
                  'y2': y2,
                  'ry2': ry2,
                  'x3': x3,
                  'y3': y3,
                  'ry3': ry3,
                  'name': key,
                  'ind': ind,
                  'r3': raise_extra(ri,x3),
                  'd3':decay[0],
                  's/n' : s_n,
                'rs/n' : rs_n
                    }])
                # dfs.append(df)
                # df = pd.DataFrame([{
                # # 's/n':snr_,
                # 'w1+w2 (ms)':w1_2/5,
                # 'w3+w1+w2 (ms)':w1_2_3/5,
                # 'w3 (ms)':w3/5,
                # 'h1 (mv)':h1/1000,
                # 'h2 (mv)':h2/1000,
                # 'h1/h2':h1_h2,
                # 'h3/h1':h3_h1,
                # 'h3/h2':h3_h2,
                # 'h3 (mv)':h3/1000,
                # 'x_chp2':x_chp2,
                # 'y_chp2':y_chp2/1000,
                # 'x_max': x1,
                # 'y_max': y1/1000,
                # 'y_p (mv)': ry1/1000,
                # 'x_min': x2,
                # 'Ny_min': y2/1000,
                # 'y_n (mv)': ry2/1000,
                # 'x_min2': x3,
                # 'Ny_min2': y3/1000,
                # 'y_m (mv)': ry3/1000,
                # 'name': key,
                # 'ind': ind,
                # 'r3 (v/s)': raise_extra(ri,x3)*5,
                # 'd3 (v/s)':decay[0]*5,
                #   }])
                dfs.append(df)
              # plt.figure(figsize= (4,4))
              # plt.plot(ri)
              # plt.plot([x1,x2,x3,x_chp2],[ry1,ry2,ry3,y_chp2],'ro')
              # plt.xlim(800,1200)
              # plt.show()
    df_eap = pd.concat(dfs)
    return df_eap

def iap_df_generator_XG(dictionary_of_input):
    apd_cos = ['APD'+str(10*i) for i in range(1,11) ]
    dict_items = list(dictionary_of_input.items())
    dfs = []
    for item in dict_items:
            key, value = item
            print(key)
            dataset_8 = train_data_sh(0.2, seed=7, UNSEEN8=[key])
            d_8 = MYDataset(all_recordings, [key], function1, function2, test_size=0.0001, val_size=0.001, seedd=seed)
            print(len(dataset_8.intras_unseen2), len(d_8.intras_raw))
            apds = dataset_8.intras_unseen_apd
            df = pd.DataFrame()
            df[apd_cos]=apds
            df['iap_sn'] = d_8.intras_sn
            df['name']=key
            df['ind'] = df.index
            dfs.append(df)
 
    df_iap = pd.concat(dfs)
    return df_iap



def eap_df_generator_EDA(dictionary_of_input):
    dict_items = list(dictionary_of_input.items())
    dfs = []
    for item in dict_items:
            key, value = item
            print(key)
            dataset_8 = train_data_sh(0.2, seed=7, UNSEEN8=[key])
            d_8 = MYDataset(all_recordings, [key], function1, function2, test_size=0.0001, val_size=0.001, seedd=seed)
            print(len(dataset_8.extras_unseen2), len(d_8.extras_raw))
            for ind in value:
                # print(ind,key)
                i = dataset_8.extras_unseen2[ind]
                t_ = d_8.times[ind]
                ri = d_8.extras_raw[ind]
                x_ = np.argmax(ri)
                i = i - np.mean(i[0:100])
                ri = ri-np.mean(ri[0:100])

                x1s, a_  = find_peaks(i,height = np.max(i)*0.6, distance=20)
                x1 = x1s[-1]
                y1, ry1 = i[x1], ri[x1]
                x2s, a_  = find_peaks(-i[x1:x1+150],height = -np.min(i[x1:x1+150])*0.6, distance=20)
                try:
                    x2= x2s[-1] + x1
                except:
                    x2 =  np.argmin(i[x1:x1+150])+1
                y2 = i[x2]
                ry2 = ri[x2]
                x3,ry3 = eap_x3(ri,x1+400) #i , chp+3
                y3 = i[x3]
                x_chp2,ry_chp2 = eap_ch2_3(ri,x2)
                y_chp2= i[x_chp2]
                x_chp1,ry_chp1 = eap_ch1(ri,x1)
                y_chp1= i[x_chp1]
                x1, y1, ry1 = np.argmax(i), np.max(i), np.max(ri)
                x2 =  np.argmin(i[x1:x1+150])+ x1
                ry2, y2 = ri[x2], i[x2]
                h1_n = y1 - y_chp1
                h1  = ry1  - ry_chp1
                x_chp1_eq = eap_ch1_y_eq (ri,x_chp1,ry_chp1,x1,x2)
                x_chp2_eq = eap_ch2_y_eq (ri,x_chp2,ry_chp2,x1,x2)
                decay = [ri[x_chp2+100]-ry3]/(x3-x_chp2-100)
                h2 = ry_chp2 - ry2
                h2_n = y_chp2 - y2
                w3 = x3 - x_chp2
                w1_2_3 = x3 -x_chp1
                
                #two options for w
                w1 = x_chp1_eq - x_chp1
                w2 = x_chp2 -x_chp2_eq
                w1_2= x_chp2-x_chp1
                
                w1_p = x1 - x_chp1
                w2_p =  x_chp2 -x2
                w1_2_p = x2-x_chp1
        
                
                
                h3 = ry_chp2 - ry3
                h3_n = y_chp2 - y3

                h1_h2 = h1/h2
                h3_h1 = h3/h1
                h3_h2 = h3/h2
                s_n = snr(i)
                rs_n = snr(ri)
                
#                 plt.plot(ri)
#                 plt.show()
#                 plt.plot(ri)
         
#                 plt.plot([x1,x2,x_chp1,x_chp2],[ri[x1],ri[x2],ri[x_chp1],ri[x_chp2]], 'ro')
#                 plt.plot([x_chp1_eq,x_chp2_eq],[ri[x_chp1],ri[x_chp2]], 'go')
#                 plt.xlim(600,1100)
#                 # plt.plot([x1,x2,x_chp1,x_chp2,x3],[ri[x1],ri[x2],ri[x_chp1],ri[x_chp2],ri[x3]], 'ro')
#                 plt.show()

                df = pd.DataFrame([{
                    't':t_,
                    'w1':w1,
                    'w2':w2,
                    
                  'w1+w2':w1_2,
                    'w1_p':w1_p,
                    'w2_p':w2_p,
                    'w1+w2_p':w1_2_p,
                    
                  'w1+w2':w1_2,
                    
                  'w3+w1+w2':w1_2_3,
                  'w3':w3,
                  'h1':h1,
                  'h2':h2,
                'h1_n':h1_n,
                  'h2_n':h2_n,
                  'h1/h2':h1_h2,
                  'h3/h1':h3_h1,
                  'h3/h2':h3_h2,
                  'h3':h3,
                    'h3_n':h3_n,
                  'x_chp2':x_chp2,
                  'y_chp2':ry_chp2,
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
                  'r3': raise_extra(ri,x3),
                  'd3':decay[0],
                  's/n' : s_n,
                'rs/n' : rs_n
                    }])
                # dfs.append(df)
                # df = pd.DataFrame([{
                # # 's/n':snr_,
                # 'w1+w2 (ms)':w1_2/5,
                # 'w3+w1+w2 (ms)':w1_2_3/5,
                # 'w3 (ms)':w3/5,
                # 'h1 (mv)':h1/1000,
                # 'h2 (mv)':h2/1000,
                # 'h1/h2':h1_h2,
                # 'h3/h1':h3_h1,
                # 'h3/h2':h3_h2,
                # 'h3 (mv)':h3/1000,
                # 'x_chp2':x_chp2,
                # 'y_chp2':y_chp2/1000,
                # 'x_max': x1,
                # 'y_max': y1/1000,
                # 'y_p (mv)': ry1/1000,
                # 'x_min': x2,
                # 'Ny_min': y2/1000,
                # 'y_n (mv)': ry2/1000,
                # 'x_min2': x3,
                # 'Ny_min2': y3/1000,
                # 'y_m (mv)': ry3/1000,
                # 'name': key,
                # 'ind': ind,
                # 'r3 (v/s)': raise_extra(ri,x3)*5,
                # 'd3 (v/s)':decay[0]*5,
                #   }])
                dfs.append(df)
              # plt.figure(figsize= (4,4))
              # plt.plot(ri)
              # plt.plot([x1,x2,x3,x_chp2],[ry1,ry2,ry3,y_chp2],'ro')
              # plt.xlim(800,1200)
              # plt.show()
    df_eap = pd.concat(dfs)
    return df_eap


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