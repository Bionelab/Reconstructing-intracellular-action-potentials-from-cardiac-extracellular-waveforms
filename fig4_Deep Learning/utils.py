# pip install McsPyDataTools
# pip install tsaug

import os
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import sys
from fastdtw import fastdtw
from peak_finder import butter_bandpass_filter, get_peak_indices, find_poration_pulses
import pandas as pd
from scipy.signal import find_peaks,peak_widths
import os
import h5py as h5
import seaborn as sns
import glob
from scipy.signal import find_peaks,peak_widths
from sklearn.metrics import cohen_kappa_score,f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import tsaug
import pickle
import peak_finder
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,peak_widths
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from scipy.signal import savgol_filter
# from fastdtw import fastdtw
import matplotlib.gridspec as gridspec
import scipy.stats
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from joblib import dump, load
from constants import *
import math
#Takes in an numpy array of intracellular traces and returns the decay rates in a numpy array (% decay per second)
def get_decay_rates(intras):
    decay_rates = []

    for intra in intras:
        intra = np.asarray(moving_filter(intra, 20))

        max_value = max(intra[:2800])
        max_index = np.where(intra == max_value)[0][0]

        min_value = min(intra[max_index:3750])
        min_index = np.where(intra[max_index:] == min_value)[0][0] + max_index

        time_diff = (max_index - min_index) * .0002

        decay_rate = (max_value - min_value) / (time_diff)
        decay_rate = decay_rate * -100
        decay_rates.append(decay_rate)

    decay_rates = np.asarray(decay_rates)
    return decay_rates

def get_apd_errors(predicted_apds, correct_apds):
    apds_percent_errors = np.zeros((10, len(predicted_apds)))
    apds_absolute_errors = np.zeros((10, len(predicted_apds)))
    for i in range(len(predicted_apds)):
        for apd in range(10):
            correct = correct_apds[i][apd]
            predicted = predicted_apds[i][apd]

            percent_error = abs(((predicted - correct) / correct) * 100)
            absolute_error = abs(predicted - correct) * .0002
            apds_percent_errors[apd][i] = abs(percent_error)
            apds_absolute_errors[apd][i] = abs(absolute_error)
    return apds_percent_errors, apds_absolute_errors

    
def calculate_accuracy(intra_predictions, intra_correct):
  pred_apds, correct_apds = get_predicted_apds(intra_predictions[:,:,0], intra_correct[:,:])
  apds_percent_errors, apds_absolute_errors = get_apd_errors(pred_apds, correct_apds)

  new_length = len(get_percentile_data(apds_percent_errors[0], 10, 90))
  apds_percent_errors_cutoff = np.zeros([10, new_length])
  apds_absolute_errors_cutoff = np.zeros([10, new_length])

  for apd in range(10):
    apds_percent_errors_cutoff[apd] = get_percentile_data(apds_percent_errors[apd], 10, 90)
    apds_absolute_errors_cutoff[apd] = get_percentile_data(apds_absolute_errors[apd], 10, 90)
  
  return pred_apds, correct_apds, apds_percent_errors_cutoff, apds_absolute_errors_cutoff

#Takes in an numpy array of intracellular traces and returns the decay rates in a numpy array (% decay per second)
def get_decay_rates(intras):
    decay_rates = []

    for intra in intras:
        intra = np.asarray(moving_filter(intra, 20))

        max_value = max(intra[:2800])
        max_index = np.where(intra == max_value)[0][0]

        min_value = min(intra[max_index:3750])
        min_index = np.where(intra[max_index:] == min_value)[0][0] + max_index

        time_diff = (max_index - min_index) * .0002

        decay_rate = (max_value - min_value) / (time_diff)
        decay_rate = decay_rate * -100
        decay_rates.append(decay_rate)

    decay_rates = np.asarray(decay_rates)
    return decay_rates

#Takes in a 1 dimensional numpy array and a window size, and applies a moving
#average filter to the data.
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

def calculate_accuracy(intra_predictions, intra_correct):
    pred_apds, correct_apds = get_predicted_apds(intra_predictions[:,:,0], intra_correct[:,:])
    apds_percent_errors, apds_absolute_errors = get_apd_errors(pred_apds, correct_apds)

    new_length = len(get_percentile_data(apds_percent_errors[0], 10, 90))
    apds_percent_errors_cutoff = np.zeros([10, new_length])
    apds_absolute_errors_cutoff = np.zeros([10, new_length])

    for apd in range(10):
        apds_percent_errors_cutoff[apd] = get_percentile_data(apds_percent_errors[apd], 10, 90)
        apds_absolute_errors_cutoff[apd] = get_percentile_data(apds_absolute_errors[apd], 10, 90)

    # return pred_apds, correct_apds, apds_percent_errors_cutoff, apds_absolute_errors_cutoff
    return pred_apds, correct_apds, apds_percent_errors, apds_absolute_errors


#Takes in a numpy array representing a single intracellular trace (4000 points), and
#two_peak_cutoff, which is a point after the action potential but before a potential second
#peak. Returns the apd widths, heights (y values of where they occur in action potential), and
#left values (x values of where apd starts in action potential)
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

#Given numpy arrays of both the prediced traces and the actual traces, returns 
#numpy arrays of the 10 APDs taken from those traces
def get_predicted_apds(predicted_traces, actual_traces):  
  pred_widths_test = []
  raw_widths_test = []
  for pred_test,raw_test in zip(predicted_traces,actual_traces):
    w1_test, h1_test, l1_test = get_apds(pred_test, 2800)
    pred_widths_test.append(w1_test)
    w2_test, h2_test, l2_test = get_apds(raw_test, 2800)
    raw_widths_test.append(w2_test)
  pred_widths = np.asarray(pred_widths_test)
  raw_widths = np.asarray(raw_widths_test)
  return pred_widths, raw_widths

# for training
def zero_maker2 (a,b,c ):
    dw_real_training = np.zeros((a.shape[0],a.shape[1]))
    dw_real_val = np.zeros((b.shape[0],b.shape[1]))
    dw_real_unseen = np.zeros((c.shape[0],c.shape[1]))
    print(dw_real_training.shape)
    print(dw_real_val.shape)
    print(dw_real_unseen.shape)
    return dw_real_training , dw_real_val,dw_real_unseen

# return train and val based on key name
def data_prep (data, all_train,val):
    trains_names = [i for i in all_train if i not in val]
    
    intras_train = []
    extras_train = [] 
    intras_val = []
    extras_val = []
    
    for key in trains_names:
        intras_train.append(data[(key,'intra')])
        extras_train.append(data[(key,'extra')])

    # if len(val) > 0:
    for key in val:
        intras_val.append(data[(key,'intra')])
        extras_val.append(data[(key,'extra')])
   
    return np.concatenate(intras_train),np.concatenate(intras_val),np.concatenate(extras_train),np.concatenate(extras_val)

# return train and val in shuffle mode
def data_prep_sh (data,all_train,unseen, size = 0.2, seed_ = 42):
 
    trains_names = [i for i in all_train if i not in unseen]
    intras_train = []
    extras_train = [] 
    for key in trains_names:
        intras_train.append(data[(key,'intra')])
        extras_train.append(data[(key,'extra')])

    x =  np.concatenate(intras_train)
    y = np.concatenate(extras_train)
    print(len(x))

    intras_train, intras_val, extras_train, extras_val = train_test_split(x, y, test_size=size, random_state=seed_)
    return intras_train,  intras_val, extras_train, extras_val 

# return test extra and inra
def data_prep_test (data,test,raw=False):

    intras_test = []
    extras_test = []
    if raw == False:
        for key in test:
            intras_test.append(data[(key,'intra')])
            extras_test.append(data[(key,'extra')])
    else:
        for key in test:
            intras_test.append(data[(key,'intra_raw')])
            extras_test.append(data[(key,'extra_raw')])
    intras =  np.concatenate(intras_test)
    extras = np.concatenate(extras_test)

    return intras,extras


def function1_min_0(intra):
    min_val = min(intra[1000:])
    span = max(intra) - min_val
    return [(val - min_val) / span for val in intra]



def data_prep (all_train,val, unseen):
    trains_names = [i for i in all_train if i not in val+unseen]
    
    intras_train = []
    extras_train = [] 
    intras_val = []
    extras_val = []
    
    for key in trains_names:
        intras_train.append(data[(key,'intra')])
        extras_train.append(data[(key,'extra')])

    # if len(val) > 0:
    # print('val',val)
    for key in val:
        # print('k',key)
        intras_val.append(data[(key,'intra')])
        extras_val.append(data[(key,'extra')])
   
    return np.concatenate(intras_train),np.concatenate(intras_val),np.concatenate(extras_train),np.concatenate(extras_val)




def model_plotter(model,extras_test2 ,intras_test2,apds_test,lens1,pol1,lens2,pol2, name, smoothing, ph ):

    intra_pred1 = model.predict(extras_test2)
    if ph == True:
        preds_ = intra_pred1[0].reshape(-1,8000)
    else:
        preds_ = intra_pred1.reshape(-1,8000)
    preds = preds_.copy()
    
    if  smoothing == True:
        preds[:,:1050] = savgol_filter(preds_[:,:1050], lens1, pol1,axis=1)
        preds[:,1000:] = savgol_filter(preds_[:,1000:], lens2, pol2,axis=1)

    apds_test_pred = get_all_apds(preds)
    print(apds_test_pred.shape) 
    apds_error_abs = np.abs(apds_test_pred- apds_test) # for test set
    apd_error_mean = np.mean(apds_error_abs, axis = 1)
    mae = np.mean(np.abs(preds-  intras_test2), axis=1)
    apds_error_reg = apds_test_pred- apds_test # for test set
    apds_error_abs = np.abs(apds_test_pred- apds_test) # for test set
    nn_apds_error_reg_test = pd.DataFrame(apds_error_reg)/5000  # for test set
    nn_apds_error_abs_test = np.abs( nn_apds_error_reg_test) # for test set
    nn_apds_error_per_test =nn_apds_error_abs_test *100/(apds_test/5000 )
    print(nn_apds_error_per_test.mean(axis = 1))
    print(nn_apds_error_abs_test.mean(axis = 1))
    fig = plt.figure(figsize=(20, 7), dpi=250)
    color = sns.color_palette("Set1")
    gs = gridspec.GridSpec(3, 6, figure=fig)  # Create 3x6 grid
    # Arrays for the titles of each column
    titles_mae = ['Lowest MAEs', 'Mid MAEs', 'Highest MAEs']
    titles_apd = ['Lowest APDs Error', 'Mid APDs Errors', 'Highest APDs Errors']
    mid0 = round(len(extras_test2)/2)
    # Arrays for the data (here I'm just using your sorting methods as placeholders)
    preds_mae = preds[np.argsort(mae)][[0,1,2,mid0-1,mid0,mid0+1,-1,-2,-3]]
    actuals_mae = intras_test2[np.argsort(mae)][[0,1,2,mid0-1,mid0,mid0+1,-1,-2,-3]]
    preds_apd = preds[np.argsort(nn_apds_error_abs_test.mean(axis = 1))][[0,1,2,mid0-1,mid0,mid0+1,-1,-2,-3]]
    actuals_apd = intras_test2[np.argsort(nn_apds_error_abs_test.mean(axis = 1))][[0,1,2,mid0-1,mid0,mid0+1,-1,-2,-3]]
    # Define the time range
    time = np.arange(1, 1601, 0.2)
    
    fig = plt.figure(figsize=(20, 7), dpi=250)
    color = sns.color_palette("Set1")
    gs = gridspec.GridSpec(3, 6, figure=fig)  # Create 3x6 grid
    for i in range(3):
        for j in range(6):
            ax = fig.add_subplot(gs[i, j])  # Add a subplot in the current grid position
            ax.label_outer()  # Only show outer tick labels
            if j < 3:  # First 3 columns are for MAEs
                ax.plot(time, actuals_mae[3*j+i], label='Actual', linestyle='-', color=color[1],linewidth=3)
                ax.plot(time,  preds_mae[3*j+i], label='Predicted', linestyle='-.', color=color[0],linewidth=3)
                if i == 0:  # Add the title on the first row
                    ax.set_title(titles_mae[j], fontsize=18)
            else:  # Last 3 columns are for APD Errors
                ax.plot(time, actuals_apd[3*(j-3)+i], label='Actual', linestyle='-', color=color[1] ,linewidth=3)
                mean = preds_apd[3*(j-3)+i]
                # std = test_preds_std[3*(j-3)+i]
                ax.plot(time, mean, label='Predicted', linestyle='-.', color=color[0],linewidth=3)
                # ax.fill_between(time, mean-std, mean+std, color=color[0], alpha=0.3)
                if i == 0:  # Add the title on the first row
                    ax.set_title(titles_apd[j-3], fontsize=18)
            if j == 0 and i == 1 :  # Add y-label on the first column of each set
                ax.set_ylabel('Normalized Voltage', fontsize=18)
            if i == 2  :  # Add x-label on the last row
                ax.set_xlabel('Time (ms)', fontsize=18)
    
    # Adding a legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from any subplot (they're the same)
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=18, bbox_to_anchor=(0.52, 1.1))  # Add a single legend for the whole figure
    
    plt.tight_layout()
    plt.savefig('results/'+name+"_pred.svg", format="svg",  dpi = 350)
    plt.show()


def get_all_apds(all_intras):
  APD_widths=[]
  for i in range(len(all_intras)):
    intra = all_intras[i]
    w1, h1, l1 = get_apds(intra, round(len(intra) * 0.7))
    APD_widths.append(w1)
  APD_widths = np.asarray(APD_widths)
  all_apds = APD_widths
  return all_apds



def model_plotter_apd(model,extras_test2 ,intras_test2,name_, ph ):

    intra_pred1 = model.predict(extras_test2)
    if ph == True:
        preds_ = intra_pred1[0].reshape(-1,8000)
    else:
        preds_ = intra_pred1.reshape(-1,8000)
    preds = preds_.copy()
    print(preds.shape)

    color = sns.color_palette("Set1")
    # Arrays for the titles of each column
    mid0 = round(len(extras_test2)*0.25)
    mid1 = round(len(extras_test2)*0.5)
    mid2 = round(len(extras_test2)*0.75)
    # [[0,mid0,mid0+1,mid1,mid1+1,mid2,mid2+1,-1]]
    preds_mae = preds[[1,-2]]
    intras = intras_test2[[1,-2]]
    extras = extras_test2[[1,-2]]
    n_cols = len([1,-2])
    print(n_cols)
    time = np.arange(1, 1601, 0.2)
    fig = plt.figure(figsize=((12/5)*n_cols, 4), dpi=250)
    color = sns.color_palette("Set1")
    gs = gridspec.GridSpec(2, n_cols, figure=fig)  # Create 3x6 grid
    for i in range(n_cols):
        print(preds_.shape)

        ax = fig.add_subplot(gs[0, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,preds_mae[i], label='Actual', linestyle='-', color=color[1],linewidth=3)
        ax.plot(time,  intras[i], label='Predicted', linestyle='-.', color=color[0],linewidth=3)

        ax = fig.add_subplot(gs[1, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,  extras[i], label='Predicted', linestyle='-.', color='black',linewidth=1)

    plt.tight_layout()
    plt.savefig("results/"+name_+"_apd_range_preds.svg", format="svg",  dpi = 350)
    plt.show()


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
def act_vs_pred_plot(x_values, y_values,name):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x_values, y_values, s=0.4, alpha=0.01)
    scatter.set_rasterized(True)  # Rasterize the scatter plot
    line_range = [min(min(x_values), min(y_values)), max(max(x_values), max(y_values))]
    plt.plot(line_range, line_range, color='red', linestyle='--')  # Red dashed line for contrast
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation = correlation_matrix[0, 1]
    plt.text(0.95, 0.05, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Scatter Plot with x = y Line and Correlation')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.grid(True)
    plt.savefig("results/"+name+"_act_vs_preds.svg", format="svg", dpi=450)
    plt.show()


def apd_comp_plot(apds_pred,apds_test,apds_pred_train,apds_train,name):
        error =  apds_pred - apds_test
        error_train = (apds_pred_train- apds_train)
        df = np.abs(error)/5000
        df2 = pd.DataFrame()
        df2[["APD "+str(10*i) for i in range(1,11)] ]= df
        
        df_train =np.abs(error_train)/5000
        df_train2 = pd.DataFrame()
        df_train2[["APD "+str(10*i) for i in range(1,11)] ]= df_train
        
        colors_green= LinearSegmentedColormap.from_list("MyCmapNameBlue", ["#000080", "#000080"])
        colors_purple = LinearSegmentedColormap.from_list("MyCmapNameOrange", ["#FF4500","#FF4500"])
        
        colors_text = ["#000080", "#FF4500"]  # Solid colors for text
        fig, ax = plt.subplots(figsize=(5, 3), dpi=250)
        
        for i, column in enumerate(df2.columns):
            c = 0
            for df_, colors, offset, label, color_text in zip([ df2,df_train2], [colors_green, colors_purple], [-0.2, 0.2], ['Test', 'Train'], colors_text):
                parts = ax.violinplot(df_[column], positions=[i+offset], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors(i/10))
                    # pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        
                bp = ax.boxplot(df_[column], positions=[i+offset], notch=True, patch_artist=True, zorder=10, widths=0.3, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white') 
                    patch.set_edgecolor('black')
                for whisker in bp['whiskers']:
                    whisker.set(color='black', linestyle='-')
                for cap in bp['caps']:
                    cap.set(color='black', linestyle='-')
                for median in bp['medians']:
                    median.set(color='black', linestyle='-')
                median_value = np.mean(df_[column])
                std_ = np.std(df_[column])
        
                c = c+1
        ax.set_xticks(np.arange(len(df2.columns)))
        xticks = ax.get_xticklabels()
        for tick in xticks:
            tick.set_color('black')
        ax.set_xticklabels(df2.columns, rotation=0, ha='center', fontsize=4)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.title.set_color('black')
        
        ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
        ax.set_title("Distribution of Errors in APDs Predicted by XGBoost", fontsize=14, color='black')
        
        ax.tick_params(axis='y', labelsize=4)
        
        legend_elements = [
                           Patch(facecolor=colors_purple(0.5), edgecolor='black', alpha=0.7, label='Test'),
                          Patch(facecolor=colors_green(0.5), edgecolor='black', alpha=0.7, label='Train')]
        # ax.legend(handles=legend_elements, loc='upper left', fontsize = 16)
        ax.legend(handles=legend_elements, loc='upper left', fontsize = 4)
        
        plt.tight_layout()
        plt.savefig("results/"+name+"apds_test_train.svg", format="svg",  dpi = 350)
        plt.show()

def mae_error( error,error_train, name_='MAE'):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    sum_error_test = error
    sum_error_train = error_train

    print('test:',sum_error_test.mean(),'+-',sum_error_test.std())
    print('train:',sum_error_train.mean(),'+-',sum_error_train.std())

    colors_text = ["#000080", "#FF4500"]  # Solid colors for text
    fig, ax = plt.subplots(figsize=(2, 2), dpi=250)
    
    for counter, data in enumerate([sum_error_train,sum_error_test]):
        parts = plt.violinplot(data, positions=[counter], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            # pc.set_facecolor(colors(i/10))
            # pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    
        bp = ax.boxplot(data, positions=[counter], notch=True, patch_artist=True, zorder=10, widths=0.3, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white') 
            patch.set_edgecolor('black')
        for whisker in bp['whiskers']:
            whisker.set(color='black', linestyle='-')
        for cap in bp['caps']:
            cap.set(color='black', linestyle='-')
        for median in bp['medians']:
            median.set(color='black', linestyle='-')
    

        median_value = np.mean(data)
        std_ = np.std(data)

    plt.title(name_)
    plt.savefig("results/"+name_+"all_error.svg", format="svg",  dpi = 350)
        
  


import jenkspy
def screen_noise_noiselevel(data_channel_raw, lowcut,highcut, fs,order):
        
        data_channel = butter_bandpass_filter(data_channel_raw, lowcut,highcut, fs,order=order)
        nois_channel = butter_bandpass_filter(data_channel_raw,highcut+0.001,fs/2.-1,fs)
        noise_level = np.std(butter_bandpass_filter(data_channel,0.8*fs/2.,fs/2.-1,fs))
        return data_channel,nois_channel,noise_level

def get_window_with_noise(data_channel_filtered,data_channel,noise_channel,p,n_points= 850):
            p_data = np.argmax(data_channel_filtered[round(p)-n_points:round(p)+(8000-n_points)]) + round(p)-n_points
            extra_segmented =data_channel[round(p_data)-n_points:round(p_data)+(8000-n_points)]
            extra_segmented_filtered =data_channel_filtered[round(p_data)-n_points:round(p_data)+(8000-n_points)]
            
            extra_noise_segmented  = noise_channel[round(p_data)-n_points:round(p_data)+(8000-n_points)]
            return extra_segmented,extra_segmented_filtered,extra_noise_segmented

def get_info(extra_segmented,extra_noise_segmented,noise_level):
        sp_0,npower_0,sn_0 = singal_noise_power(extra_segmented,extra_noise_segmented)
        width_extras = peak_widths(extra_segmented,peaks=[np.argmax(extra_segmented)],rel_height=0.5)[0][0]
        return sp_0,npower_0,sn_0 ,width_extras


def sn_ratio (extra_segmented):
    lens2 = 1001
    pol1 = 3
    span0 = span(extra_segmented)
    seg_filterd = savgol_filter(extra_segmented[1000:], lens2, pol1)
    std_ = np.std(extra_segmented[1000:]-seg_filterd)
    snr = span0/std_
    # plt.plot(extra_segmented[1000:])
    # plt.plot(seg_filterd)
    # plt.show()
    return std_,span0,snr

def get_width(extra_segmented):
    width_extras = peak_widths(extra_segmented,peaks=[np.argmax(extra_segmented)],rel_height=0.5)[0][0]
    return width_extras

def span(x):
    return np.max(x)-np.min(x)

def smoother(preds_,lens1= lens1,pol1 = pol1,lens2 = lens2,pol2 = pol2):
        preds = preds_.reshape(-1,8000).copy()
        preds[:,:1050] = savgol_filter(preds_[:,:1050], lens1, pol1,axis=1)
        preds[:,1000:] = savgol_filter(preds_[:,1000:], lens2, pol2,axis=1)
        return preds

def eap_x3(i,x_ch2):
      window_length, poly_order = 75,5
      sdata = savgol_filter(i[x_ch2+300:] , window_length, poly_order)
      sdata = savgol_filter(sdata, window_length, poly_order)
      x3 = np.argmin([sdata]) + x_ch2+300
      y3 = np.min([sdata])
      return x3,y3

def eap_ch2_3(i,x2):
    window_length, poly_order = 75,3
    i_n2 = i[x2:x2+300].copy()
    sdata2 = savgol_filter(i_n2[:] , window_length, poly_order)
    # sdata2 = savgol_filter(sdata2, window_length, poly_order)
    
    sdata_g2 = np.gradient(sdata2)
    sdata_g2 = savgol_filter(sdata_g2, window_length, poly_order)
    ch_p2 = detect_changes_eap2(sdata_g2[10:-10])+10
    x_ch2 = ch_p2+x2
    y_ch2 = sdata2[ch_p2]
    return x_ch2, y_ch2
def detect_changes_eap2(signal):
    breaks = jenkspy.jenks_breaks(signal, n_classes=2)
    change_points = [i for i in range(1, len(signal)) if breaks[0] <= signal[i] < breaks[1]]
    return change_points[0]



def std_over_decay(eap, x_ch2,x3):
    window_length, poly_order = 75,3
    i_n2 = eap[x_ch2:x3].copy()
    sdata2 = savgol_filter(i_n2 , window_length, poly_order)
    sdata2 = savgol_filter(sdata2, window_length, poly_order)
    std = np.std(i_n2-sdata2)
    span_ = np.max(sdata2)-np.min(sdata2)
    return span_,std


def predictor(model1,list_of_data):
    
    pred1=model1.predict(list_of_data)[0].reshape(-1,8000)
    pred1_smooth = smoother(pred1.reshape(-1,8000))
    return pred1,pred1_smooth


def find_var_name(var, local_vars):
    return [name for name, value in local_vars.items() if value is var]


def variation(extra_segmented):
    lens2 = 301
    pol1 = 3
    seg_filterd = savgol_filter(extra_segmented[1000:], lens2, pol1)
    var_filtered = np.std(seg_filterd)
    
    diff = extra_segmented[1000:]-seg_filterd
    var_diff = np.std(diff)
    rel_var = var_filtered/var_diff

    
    return var,var_diff,rel_var


def plot_errorss( data_feature_test,error_name,name_, col1):

    APD_ER = ['APD_ER_'+str(10*i) for i in range(1,11)]
    APD_PER = ['APD_PER_'+str(10*i) for i in range(1,11)]
    plt.style.use('default')
    plt.rcParams.update({'font.size': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    error_target =error_name
    num_cols = len(col1)//2
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows), dpi=250,)
    cmap = plt.get_cmap('viridis', num_rows+2)
    for i_idx in range(num_rows):
        for j_idx in range(num_cols):
            i = col1[i_idx * num_cols + j_idx]
            print(i)
            df_feat_test90 = data_feature_test
            df_feat_test90 = df_feat_test90[(df_feat_test90['h1']>0.1)&
                                            (df_feat_test90['h3']<0.05)&
                                            (df_feat_test90['r3']>df_feat_test90['r3'].min())
                                           ]
            correlation_matrix = np.corrcoef(df_feat_test90[error_target], df_feat_test90[i])
            correlation_xy = correlation_matrix[0, 1]
            # Plot on the appropriate subplot
            ax = axs[i_idx, j_idx]
            ax.scatter(y=df_feat_test90[error_target], x= df_feat_test90[i], s=5, color=cmap(i_idx))
            formatter = ScalarFormatter(useMathText=True)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
            ax.yaxis.set_ticks_position('left')
            if j_idx == 0:  # First column
      
                error_target2=error_target
                ax.set_ylabel(error_target2)
               
            ax.set_xlabel(i)
            if j_idx != 0:
                ax.yaxis.set_tick_params(width=0.5, length=5, which='both', direction='in', labelleft=False)
    
            ax.text(0.1, 0.95, f'r: {correlation_xy:.2f}', horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, fontsize=10, color='red',)
    
    fig.set_facecolor('white')
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    plt.savefig('results/'+name_+error_name+"_nn_errors.svg", format="svg",  dpi = 250)
    plt.show()

def act_vs_pred_plot(x_values, y_values,name):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x_values, y_values, s=0.4, alpha=0.01)
    scatter.set_rasterized(True)  # Rasterize the scatter plot
    line_range = [min(min(x_values), min(y_values)), max(max(x_values), max(y_values))]
    plt.plot(line_range, line_range, color='red', linestyle='--')  # Red dashed line for contrast
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation = correlation_matrix[0, 1]**2
    plt.text(0.95, 0.05, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Scatter Plot with x = y Line and Correlation')
    plt.xlabel('ActualV alues')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig("results/"+name+"_act_vs_preds.svg", format="svg", dpi=450)
    plt.show()

def apd_comp_plot(apds_pred,apds_test,apds_pred_train,apds_train,name):
        error =  apds_pred - apds_test
        error_train = (apds_pred_train- apds_train)
        df = np.abs(error)/5000
        df2 = pd.DataFrame()
        df2[["APD "+str(10*i) for i in range(1,11)] ]= df
        
        df_train =np.abs(error_train)/5000
        df_train2 = pd.DataFrame()
        df_train2[["APD "+str(10*i) for i in range(1,11)] ]= df_train
        
        colors_green= LinearSegmentedColormap.from_list("MyCmapNameBlue", ["#000080", "#000080"])
        colors_purple = LinearSegmentedColormap.from_list("MyCmapNameOrange", ["#FF4500","#FF4500"])
        
        colors_text = ["#000080", "#FF4500"]  # Solid colors for text
        fig, ax = plt.subplots(figsize=(5, 3), dpi=250)
        
        for i, column in enumerate(df2.columns):
            c = 0
            for df_, colors, offset, label, color_text in zip([ df_train2,df2], [colors_green, colors_purple], [-0.2, 0.2], ['Test', 'Train'], colors_text):
                parts = ax.violinplot(df_[column], positions=[i+offset], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors(i/10))
                    # pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        
                bp = ax.boxplot(df_[column], positions=[i+offset], notch=True, patch_artist=True, zorder=10, widths=0.3, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white') 
                    patch.set_edgecolor('black')
                for whisker in bp['whiskers']:
                    whisker.set(color='black', linestyle='-')
                for cap in bp['caps']:
                    cap.set(color='black', linestyle='-')
                for median in bp['medians']:
                    median.set(color='black', linestyle='-')
                median_value = np.mean(df_[column])
                std_ = np.std(df_[column])
        
                c = c+1
        ax.set_xticks(np.arange(len(df2.columns)))
        xticks = ax.get_xticklabels()
        for tick in xticks:
            tick.set_color('black')
        ax.set_xticklabels(df2.columns, rotation=0, ha='center', fontsize=4)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.title.set_color('black')
        
        ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
        ax.set_title("Distribution of Errors in APDs Predicted by XGBoost", fontsize=14, color='black')
        
        ax.tick_params(axis='y', labelsize=4)
        
        legend_elements = [
                           Patch(facecolor=colors_purple(0.5), edgecolor='black', alpha=0.7, label='Test'),
                          Patch(facecolor=colors_green(0.5), edgecolor='black', alpha=0.7, label='Train')]
        # ax.legend(handles=legend_elements, loc='upper left', fontsize = 16)
        ax.legend(handles=legend_elements, loc='upper left', fontsize = 4)
        
        plt.tight_layout()
        plt.savefig("results/"+name+"apds_test_train.svg", format="svg",  dpi = 350)
        # plt.show()

def model_plotter_apdq(preds ,extras_test2,intras_test2,name_, ph ):
    print(preds.shape)

    color = sns.color_palette("Set1")
    # Arrays for the titles of each column
    mid0 = round(len(intras_test2)*0.25)
    mid1 = round(len(intras_test2)*0.5)
    mid2 = round(len(intras_test2)*0.75)
    # [[0,mid0,mid0+1,intras_test2,mid1+1,mid2,mid2+1,-1]]
    preds_mae = preds[[1,-2]]
    intras = intras_test2[[1,-2]]
    extras = extras_test2[[1,-2]]
    n_cols = len([1,-2])
    print(n_cols)
    time = np.arange(1, 1601, 0.2)
    fig = plt.figure(figsize=((12/5)*n_cols, 4), dpi=250)
    color = sns.color_palette("Set1")
    gs = gridspec.GridSpec(2, n_cols, figure=fig)  # Create 3x6 grid
    for i in range(n_cols):
        print(preds.shape)

        ax = fig.add_subplot(gs[0, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,preds_mae[i], label='Actual', linestyle='-', color=color[1],linewidth=3)
        ax.plot(time,  intras[i], label='Predicted', linestyle='-.', color=color[0],linewidth=3)

        ax = fig.add_subplot(gs[1, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,  extras[i], label='Predicted', linestyle='-.', color='black',linewidth=1)

    plt.tight_layout()
    plt.savefig("results/"+name_+"_apd_range_preds.svg", format="svg",  dpi = 350)
    plt.show()


def variation(extra_segmented):
    lens2 = 301
    pol1 = 3
    seg_filterd = savgol_filter(extra_segmented[1000:], lens2, pol1)
    var_filtered = np.std(seg_filterd)
    
    diff = extra_segmented[1000:]-seg_filterd

    var_diff = np.std(diff)
    rel_var = var_filtered/var_diff

    
    return var_filtered,var_diff,rel_var

from scipy.signal import butter,filtfilt
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

def noise_level (data, fs = 5000):
    return np.std(butter_bandpass_filter(filtered_data[i],0.8*fs/2.,fs/2.-1,fs))#get high frequency component

# for both iap and eap
def eap_window_maker2(fs,lowcut,lowcut2,order,highcut1,highcut2, f2, ch_number,lm_baseline ):
        extra1 = f2[ch_number][lm_baseline[0]-4000:]
        f_extra1 = butter_bandpass_filter(extra1, lowcut, highcut1, fs,order=order) # filtered singal
        n_extra1 = butter_bandpass_filter(extra1,highcut1+0.001,fs/2.-1,fs, order=order)#get high frequency component noise 
        f_extra2 = butter_bandpass_filter(extra1, lowcut2, highcut2, fs,order=order) # filtered singal 
        n_extra2 = butter_bandpass_filter(extra1,0.8*(fs/2),fs/2.-1,fs, order=order)#get high frequency component noise 

        fw1s=[]
        fw2s=[]
        nw1s=[]
        nw2s=[]
        rw1s=[]
    
        for n0 in range(0,len(lm_baseline)):
            # print(lm_baseline[n0])
            t_ = lm_baseline[n0]
            # print(t_)
            r_w1 = extra1[t_-1000:t_+7000]
            rw1s.append(r_w1)
            f_w1 = f_extra1[t_-1000:t_+7000]
            fw1s.append(f_w1)
            f_w2 = f_extra2[t_-1000:t_+7000]
            fw2s.append(f_w2)
            n_w1 = n_extra1[t_-1000:t_+7000]
            nw1s.append(n_w1)
            n_w2 = n_extra2[t_-1000:t_+7000]
            nw2s.append(n_w2)

        return rw1s,fw1s,fw2s,nw1s,nw2s # raw, only trend filtered, trend and noise filtered, nosie, peaks time or time points
def singal_prep2(data, fs,lowcut,highcut,order, buffer_time = 200000, resample = True ):

    if resample == True: # for nea
        data2 = signal.resample_poly(data, up=2, down=1) 
    else:
        data2=data

    p1_b =butter_bandpass_filter(data2, lowcut2, highcut2, fs,order=order) # filtered singal but only for long trend, we kept the noise?
    noise1_b = butter_bandpass_filter(data2,0.8*(fs/2),fs/2.-1,fs, order=order)#get high frequency component noise 
    p2_b = p1_b[np.argmax(np.abs(p1_b))+buffer_time:]
    noise2_b = noise1_b[np.argmax(np.abs(p1_b))+buffer_time:]
    # p3_b = p2_b - np.mean(p2_b)
    p3_b = p2_b
    
    return p1_b,noise2_b,p3_b  
    # singal filtered, raw, singal after the elecroporatiion, noise after electroporation, method b is csaba



def baseline(data,i , t, fs,lowcut,order,highcut,ps, ds):
    f_data = butter_bandpass_filter(data[i][t[0]:t[1]], lowcut, highcut, fs,order=order) # filtered singal
    f_data = f_data - np.mean(f_data)
    peaks, __ = find_peaks(f_data, prominence=ps, distance=ds) # for extras
    plt.figure(figsize = (40,2))
    plt.plot(f_data)
    plt.plot(peaks,f_data[peaks],'ro')
    plt.show()
    return peaks

def function2(extra):
  std = np.std(extra[1150:1350])
  return ((extra-np.mean(extra))/(std * 60))


def singal_noise_power(spike, noise):
    sp_ = np.mean(np.array(spike) ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(np.array(noise) ** 2)  # Square of the signal's amplitude
    sn_ = 10 * math.log10(sp_ / npower_)

    return sp_,npower_,sn_


def act_vs_pred_plot_quantile(x_values, y_values, name='q'):
    plt.figure(figsize=(10, 10))

    # Calculate median and confidence interval
  
    median_y = y_values[1]
    lower_ci = y_values[0]
    upper_ci = y_values[2]

    print(x_values.shape,median_y.shape,lower_ci.shape,upper_ci.shape)
    # Plot median line and CI area
    # plt.plot(x_values, median_y, color='blue', label='Median')
    

    # Scatter plot for actual points
    
    
    scatter_lower = plt.scatter(x_values, lower_ci, s=0.2, alpha=0.01, color='green')
    scatter_median = plt.scatter(x_values, median_y, s=0.2, alpha=0.01, color='red')
    scatter_upper = plt.scatter(x_values, upper_ci, s=0.2, alpha=0.01, color='blue')
    
    # Set rasterized to True for all scatter plots
    scatter_lower.set_rasterized(True)
    scatter_upper.set_rasterized(True)
    scatter_median.set_rasterized(True)

    # Additional elements (line, text, labels)
    print(y_values[0].shape)
    print(np.min(y_values[0]))
    print(np.min(x_values))
    print(np.max(y_values[2]))
    print(np.max(x_values))
    # line_range = [min(np.min(x_values), np.min(y_values[0])), max(np.max(x_values), np.max(y_values[2]))]
    line_range = [0,1]
    plt.plot(line_range, line_range, color='black', linestyle='--')
    correlation = np.corrcoef(x_values, y_values[1])[0, 1]
    plt.text(0.95, 0.05, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Scatter Plot with Median and Confidence Interval')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/"+name+"_act_vs_preds.svg", format="svg", dpi=450)
    plt.show()


def smoother(preds_,lens1= lens1,pol1 = pol1,lens2 = lens2,pol2 = pol2):
        preds = preds_.reshape(-1,8000).copy()
        preds[:,:1050] = savgol_filter(preds_[:,:1050], lens1, pol1,axis=1)
        preds[:,1000:] = savgol_filter(preds_[:,1000:], lens2, pol2,axis=1)
        return preds

def get_all_apds(all_intras):
  APD_widths=[]
  for i in range(len(all_intras)):
    intra = all_intras[i]
    w1, h1, l1 = get_apds(intra, round(len(intra) * 0.7))
    APD_widths.append(w1)
  APD_widths = np.asarray(APD_widths)
  all_apds = APD_widths
  return all_apds

def return_apds(df,dic_eaps):
    preds1 = np.array([dic_eaps[(i,'p1_smooth')] for i in df.index])
    preds2 = np.array([dic_eaps[(i,'p2_smooth')] for i in df.index])
    preds3 = np.array([dic_eaps[(i,'p3_smooth')] for i in df.index])
    apd1 = get_all_apds(preds1.reshape(-1,8000))
    print('ap1 done')
    apd2 = get_all_apds(preds2.reshape(-1,8000))
    print('ap2 done')
    apd3 = get_all_apds(preds3.reshape(-1,8000))
    print('ap3 done')
    
    apd_col1 = ['APD_P1_'+str(i) for i in range(1,11)]
    apd_col2 = ['APD_P2_'+str(i) for i in range(1,11)]
    apd_col3 = ['APD_P3_'+str(i) for i in range(1,11)]
    df2 = df.copy()
    df2.loc[df2.index,apd_col1] = apd1
    df2.loc[df2.index,apd_col2] = apd2
    df2.loc[df2.index,apd_col3] = apd3
    # to screen wrong predictions of apd
    for cols in ['APD_P1_9','APD_P3_7','APD_P1_5']:
        df2 = df2[df2[cols]>2000]
    return apd1, apd2, apd3,df2


def all_channels_apd_plotters(df3):
    colors = ['#F68B1F','#6DC8BF','#92278F']
    darker_colors = [darken_color(c) for c in colors]

    med  = 'APD_P2_'+ str(5)
    high = 'APD_P3_'+ str(5)
    low = 'APD_P1_'+ str(5)
    
    med2  = 'APD_P2_'+ str(7)
    high2 = 'APD_P3_'+ str(7)
    low2 = 'APD_P1_'+ str(7)

    med3  = 'APD_P2_'+ str(9)
    high3 = 'APD_P3_'+ str(9)
    low3= 'APD_P1_'+ str(9)

    fig, axes = plt.subplots(7, 7, figsize=(14, 14), dpi=200)
    axes = axes.flatten() 
    df3['p2']= df3['p']/(5000*60)
    for i, ch in enumerate(df3['ch'].unique()):
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            yerr = [df_ch[med]/5000 - df_ch[low]/5000, df_ch[high]/5000 - df_ch[med]/5000]
            yerr2 = [df_ch[med2]/5000 - df_ch[low2]/5000, df_ch[high2]/5000 - df_ch[med2]/5000]
            yerr3 = [df_ch[med3]/5000 - df_ch[low3]/5000, df_ch[high3]/5000 - df_ch[med3]/5000]
            axes[i].errorbar(df_ch['p2'], df_ch[med]/5000, yerr=yerr, fmt='o', color=colors[0], ecolor=darker_colors[0], markersize=0.3, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med2]/5000, yerr=yerr2, fmt='o', color=colors[1], ecolor=darker_colors[1], markersize=0.3, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med3]/5000, yerr=yerr3, fmt='o', color=colors[2], ecolor=darker_colors[2], markersize=0.3, alpha=0.5)

            y_min, y_max = axes[i].get_ylim()
            axes[i].set_yticks([0.3, 1.3])
            axes[i].set_ylim([0.3,1.3])
            for spine in axes[i].spines.values():
                spine.set_edgecolor('black')
            # axes[i].set_title(f'ch {int(1+ch)}')
            if i % 7 != 0:
                axes[i].set_yticklabels([])
            if i < 42:
                axes[i].set_xticklabels([])
            axes[i].set_xticks([axes[i].get_xticks()[0], axes[i].get_xticks()[-1]])
            # axes[i].set_yticks([axes[i].get_yticks()[0], axes[i].get_yticks()[-1]])

    for j in range(i + 1, 49):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/apd_splot.svg', format='svg')
    plt.show()


def all_channels_plotters(df3, t, dic_eaps, iap):
    t2 = t * 5000
    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    indicies = df3.reset_index().set_index(['p2', 'ch']).loc[t]['index'].tolist()

    for i, ch in enumerate(indicies):
        ax = axes[i]
        if iap:
            ax.plot(np.arange(t2 - 850, t2 + 8000 - 850) / (5000 * 60), dic_eaps[(ch, 'p2')], color='black')
            ax.fill_between(np.arange(t2 - 850, t2 + 8000 - 850) / (5000 * 60), dic_eaps[(ch, 'p1')], dic_eaps[(ch, 'p3')], color='red', alpha=0.5)
            ax.set_ylim([0, 1])
            name = 'iap'
        else:
            ax.plot(np.arange(t2 - 850, t2 + 8000 - 850) / (5000 * 60), dic_eaps[(ch, 'eap_raw_norm')], color='black')
            ax.set_ylim([-0.5, 0.5])
            name = 'eap'

        # Remove all x and y tick labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig('results/' + name + 'channels_preds_plot.svg', format='svg')
    plt.show()


def all_channels_apd_plotters(df3):
    colors = ['#F68B1F','#6DC8BF','#92278F']
    darker_colors = [darken_color(c) for c in colors]

    med  = 'APD_P2_' + str(5)
    high = 'APD_P3_' + str(5)
    low = 'APD_P1_' + str(5)
    
    med2  = 'APD_P2_' + str(7)
    high2 = 'APD_P3_' + str(7)
    low2 = 'APD_P1_' + str(7)

    med3  = 'APD_P2_' + str(9)
    high3 = 'APD_P3_' + str(9)
    low3 = 'APD_P1_' + str(9)

    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    df3['p2'] = df3['p'] / (5000 * 60)
    
    for i, ch in enumerate(df3['ch'].unique()):
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            yerr = [df_ch[med]/5000 - df_ch[low]/5000, df_ch[high]/5000 - df_ch[med]/5000]
            yerr2 = [df_ch[med2]/5000 - df_ch[low2]/5000, df_ch[high2]/5000 - df_ch[med2]/5000]
            yerr3 = [df_ch[med3]/5000 - df_ch[low3]/5000, df_ch[high3]/5000 - df_ch[med3]/5000]
            axes[i].errorbar(df_ch['p2'], df_ch[med]/5000, yerr=yerr, fmt='o', color=colors[0], ecolor=darker_colors[0], markersize=0.3, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med2]/5000, yerr=yerr2, fmt='o', color=colors[1], ecolor=darker_colors[1], markersize=0.3, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med3]/5000, yerr=yerr3, fmt='o', color=colors[2], ecolor=darker_colors[2], markersize=0.3, alpha=0.5)

            # Remove all x and y tick labels and ticks
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig('results/apd_splot.svg', format='svg')
    plt.show()

import matplotlib.colors as mcolors
def darken_color(color, factor=0.6):
    # Convert hex color to RGB
    rgb = mcolors.hex2color(color)
    # Darken the RGB color
    dark_rgb = [x * factor for x in rgb]
    return mcolors.to_hex(dark_rgb)

def find_in(df, target):
    differences = np.abs(df['p2'] - target)
    closest_index = df.index[np.argmin(differences)]
    return closest_index


def apd_changes_summary(df, perc= False):

    # Define colors
    box_colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, silver, bronze
    dot_color = 'black'  # Color for the individual data points
    
    # Create a boxplot
    plt.figure(figsize=(4, 4), dpi = 250)
    
    # Create box plots for each APD group
    for i, apd in enumerate(df['apd'].unique()):
        # Filter the DataFrame for the current APD group
        if perc == False:
            apd_data = df[df['apd'] == apd]['mean']/5
            y_label = 'Drug Induced APD Change (ms)'
            name ='ms'

        else:
            apd_data = df[df['apd'] == apd]['mean_perc']
            y_label = 'Drug Induced APD Change %'
            name = 'perc'
        
        # Add boxplot for the APD group
        bp = plt.boxplot(apd_data, positions=[i], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=box_colors[i % len(box_colors)], color=dot_color),
                         medianprops=dict(color='orange'),
                         whiskerprops=dict(color=dot_color),
                         capprops=dict(color=dot_color),
                         flierprops=dict(markerfacecolor=dot_color, marker='o', markersize=1))
        
        # Add scatter plot for individual data points
        plt.scatter(np.random.normal(i, 0.04, size=len(apd_data)), apd_data,
                    color=dot_color, alpha=1, edgecolor=None, zorder=2, s= 2)
    
    # Customize the plot to make it aesthetically pleasing
    plt.xticks(range(len(df['apd'].unique())), df['apd'].unique())
    plt.ylabel(y_label, fontsize=14)
    # plt.xlabel('APD Value', fontsize=14)
    
    # Hide the right and top spines
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    # plt.gca().xaxis.set_ticks_position('bottom')
    # plt.gca().yaxis.set_ticks_position('left')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/apd_box_plots'+name+'.svg', format='svg', dpi=300)
    
    # Show the plot
    plt.show()


