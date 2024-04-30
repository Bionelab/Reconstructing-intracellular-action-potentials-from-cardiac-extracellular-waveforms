#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import McsPy
import McsPy.McsData
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import iqr
from scipy.signal import correlate
get_ipython().run_line_magic('matplotlib', 'inline')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if data.ndim==1:
        y = filtfilt(b, a, data)
    else:
        y = filtfilt(b,a,data,axis=1)
    return y

#import matplotlib as mpl
#mpl.rcParams['agg.path.chunksize'] = 100000
from scipy.signal import find_peaks,peak_prominences,peak_widths
from ipywidgets import interact,IntSlider

def find_poration_pulses(filtered_rec):
    largest_events=np.max(np.abs(filtered_rec),axis=0)*np.median(np.abs(filtered_rec),axis=0)
    #some traces are very large with real signals, but not all. We can higlight the voltage pulses by weighing with median.
    thresh=200*np.median(largest_events)
    dist=len(largest_events)/100.
    #p_idx,_=find_peaks(largest_events,height=thresh,distance=dist,width=100)
    p_idx,_=find_peaks(largest_events,height=thresh,prominence=thresh,distance=dist,wlen=10000,width=(100,5000),rel_height=0.8)

    return p_idx
def get_peak_indices(filtered_rec,buffer,poration_splits,trace_idx):
    def find_rhythm(trace,shortest_beat_estimate,longest_beat_estimate):
        corr=correlate(trace,trace,mode='same')
        corr=corr[len(corr)//2:len(corr)//2+longest_beat_estimate]
        idx=np.argmax(corr[shortest_beat_estimate:])+shortest_beat_estimate
        return idx    
    slices=poration_splits
            
    ##get global rhythm per bits between poration pulses    
    if len(slices)>0:
        periods=[]
        for i in range(len(slices)-1):

            if i==0:
                sub_trace=filtered_rec[:,slices[i]:slices[i+1]-buffer]
            elif i==len(slices)-1:
                sub_trace=filtered_rec[:,slices[i]+buffer:slices[i+1]]
            else:
                sub_trace=filtered_rec[:,slices[i]+buffer:slices[i+1]-buffer]
            if slices[i+1]-slices[i]>2*buffer+1: #not an empty slice
                periods.append(find_rhythm(np.mean(sub_trace,axis=0),shortest_beat_estimate=500,longest_beat_estimate=8000))
            else:
                periods.append(0)
    else:
        single_period=find_rhythm(np.mean(filtered_rec,axis=0),shortest_beat_estimate=500,longest_beat_estimate=8000)

    
    
    

    #buffer carve this many datapoints around the pulses
    peak_indices=[]
    trace=filtered_rec[trace_idx]
    
    if len(slices)>0:
        for i in range(len(slices)-1):

            if i==0:
                sub_trace=trace[slices[i]:slices[i+1]-buffer]
            elif i==len(slices)-1:
                sub_trace=trace[slices[i]+buffer:slices[i+1]]
            else:
                sub_trace=trace[slices[i]+buffer:slices[i+1]-buffer]
            #thresh=3*np.median(np.abs(sub_trace))/0.675
            period=0.6*periods[i]
            if slices[i+1]-slices[i]>2*buffer+1: #not an empty slice
                noise_level=np.std(butter_bandpass_filter(sub_trace,2000.,2499.,fs=5000.)) #only keep high freq to estimate noise (noise)
                peaks,_=find_peaks(np.abs(sub_trace),height=12*noise_level,distance=period) # 12 is a good trade-off. noise std smaller cause of filter, so need higher threshold
                if i!=0:
                    peaks+=slices[i]+buffer #shift number to normal datapoint
                peak_indices.append(peaks)
    else:
        sub_trace=trace
        period=0.6*single_period

        noise_level=np.std(butter_bandpass_filter(trace,2000.,2499.,fs=5000.)) #only keep high freq to estimate noise (noise)
        peaks,_=find_peaks(np.abs(sub_trace),height=12*noise_level,distance=period)
        peak_indices.append(peaks)
        #print(period)
    return np.concatenate(peak_indices)


# In[ ]:





# In[ ]:





# In[ ]:




