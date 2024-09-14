import zipfile
import os
import pyabf
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt,savgol_filter
from scipy.signal import savgol_filter
from scipy.signal import find_peaks,peak_widths
import os
from peak_finder import *
from constants import *
from matplotlib.colors import Normalize


#normalizing singals
def function1_min_0(intra):
    min_val = min(intra[:])
    span = max(intra) - min_val
    return [(val - min_val) / span for val in intra]

# butter passpand
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

def get_apd(spike):
    w1_test, h1_test, l1_test = get_apds(spike, 7000)
    pred_widths = np.asarray(w1_test)
    pred_ls= np.asarray(l1_test)
    return pred_widths, pred_ls


def ds_finder (x):
    corr = signal.correlate(x, x, mode = 'full')
    lags = signal.correlation_lags(len(x), len(x))
    corr /= np.max(corr)
    corr2 = corr[lags > 0]
    lags2 = lags[lags > 0]
    first_ind = np.where(corr2 < 0)[0][0] if np.any(corr2 < 0) else None
    ds = np.argmax(corr2[first_ind:first_ind+50000])+first_ind
    return ds/2


def custom_window(array, window_size):
    n = len(array)
    return [np.max(array[i:i+window_size]) for i in range(0, n, window_size)]

def ps_finder(x):
    x2 = x[round(len(x)/3):]
    window_size = round(ds_finder (x2))    
    ps = np.median(np.array(custom_window(x2, window_size)))/2
    return ps

def singal_prep(data, fs,lowcut,lowcut2,highcut,highcut2,order, buffer_time = 200000, resample = True):

    if resample == True:
        data2 = signal.resample_poly(data, up=2, down=1) 
    else:
        data2=data
    p1_ =butter_bandpass_filter(data2, lowcut, highcut, fs,order=order) # filtered singal but only for long trend, we 
    noise1_ = butter_bandpass_filter(data2,highcut+0.001,fs/2.-1,fs, order=order)
    p2_ = p1_[np.argmax(np.abs(p1_))+buffer_time:]
    noise2_ = noise1_[np.argmax(np.abs(p1_))+buffer_time:]
    p3_ = p2_ - np.mean(p2_)
    p1_b =butter_bandpass_filter(data2, lowcut2, highcut2, fs,order=order) # filtered singal but only for long trend, 
    noise1_b = butter_bandpass_filter(data2,0.8*(fs/2),fs/2.-1,fs, order=order)#get high frequency component noise 
    p2_b = p1_b[np.argmax(np.abs(p1_b))+buffer_time:]
    noise2_b = noise1_b[np.argmax(np.abs(p1_b))+buffer_time:]
    p3_b = p2_b - np.mean(p2_b)

    return p1_,p3_,noise2_ ,p1_b,noise2_b,p3_b  # singal filtered, raw, singal after the elecroporatiion, noise after 


def peaks_matcher2(p3_,n3_,noise_patch, noise_nea,arrythmia = False):
    
    len_min = min(len(p3_),len(n3_))
    p4_ = p3_[:len_min]
    n4_ = n3_[:len_min]
    noise_patch4 = noise_patch[:len_min]
    noise_nea4  = noise_nea[:len_min]
    
    signal1 = p4_[:]
    signal2 = n4_[:]

    corr = signal.correlate(signal1, signal2, mode = 'full')
    lags = signal.correlation_lags(len(signal1), len(signal2))
    corr /= np.max(corr)
    ind = np.argmax(corr)
    lag = lags[ind]

    if lag > 0:
        aligned_signal1 = signal1[lag:]
        aligned_signal2 = signal2
    else:
        aligned_signal1 = signal1[:]
        aligned_signal2 = signal2[abs(lag):]
        
    len_min = min(len(aligned_signal1),len(aligned_signal2))
    p5_ =  aligned_signal1[:len_min]
    n5_ =  aligned_signal2[:len_min]
        
    print(len(p5_)==len(n5_))

    if arrythmia == True:
        ds_devider =2.5
    else:
        ds_devider = 1
    p_peaks, __ = find_peaks(p4_, prominence=ps_finder(p4_)/1.2, distance=ds_finder (p4_)/ds_devider) 
    n_peaks, __ = find_peaks(n4_, prominence=ps_finder(n4_)/1.2, distance=ds_finder (n4_)/ds_devider) 
    
    start = max(p_peaks[0],n_peaks[0])
    end = min(p_peaks[-2],n_peaks[-2])
    print('start',start,'end',end)

    p5_ = p4_[start+100:end-100]
    n5_ = n4_[start+100:end-100]

    noise_patch5=noise_patch4[start+100:end-100]
    noise_nea5=noise_nea4 [start+100:end-100]

    p_peaks2, __ = find_peaks(p5_, prominence=ps_finder(p5_)/1.2, distance=ds_finder (p5_)/ds_devider) 
    n_peaks2, __ = find_peaks(n5_, prominence=ps_finder(n5_)/1.2, distance=ds_finder (n5_)/ds_devider) 
                              
    return p5_,n5_,p_peaks2,n_peaks2,noise_patch5,noise_nea5


def average_correlation(array1, array2):
    correlation_matrix = np.corrcoef(array1, array2)
    average_correlation = correlation_matrix[0, 1]
    return average_correlation


def singal_noise_power(spike, noise):
    sp_ = np.mean(np.array(spike) ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(np.array(noise) ** 2)  # Square of the signal's amplitude
    sn_ = 10 * math.log10(sp_ / npower_)

    return sp_,npower_,sn_


def ds_finder_iap_patch (x,y):
    corr = signal.correlate(x, y, mode = 'full')
    lags = signal.correlation_lags(len(x), len(y))
    corr /= np.max(corr)
    ds= lags[np.argmax(corr)]
    return ds



def singal_prep2(data, fs,lowcut,lowcut2,highcut,highcut2,order, buffer_time = 200000, resample = True ):

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
    
    p1_ =butter_bandpass_filter(data2, lowcut, highcut, fs,order=order) # filtered singal but only for long trend, we kept the noise?
    noise1_ = butter_bandpass_filter(data2,highcut+0.001,fs/2.-1,fs, order=order)
    p2_ = p1_[np.argmax(np.abs(p1_b))+buffer_time:] # i used the first filter for it too
    noise2_ = noise1_[np.argmax(np.abs(p1_b))+buffer_time:] # i used the first filter for it too
    # p3_ = p2_ - np.mean(p2_)   
    p3_= p2_
    
    return p1_,p3_,noise2_ ,p1_b,noise2_b,p3_b  # singal filtered, raw, singal after the elecroporatiion, noise after electroporation, method b is csaba


def patch_nea_dic_df (p_data,n_data,rec_names):
    
    assert len(n_data)== len(p_data)
    assert len(n_data)== len(rec_names)
    dic_patch_nea = {}
    df = pd.DataFrame()
    total_counter = 0
    for p_,n_,name in zip(p_data,n_data,rec_names):
       
        if 'arrythmia' in name:
            arrythmia = True
        else:
             arrythmia = False
        counter_each = 0
        n1_,n3_,n_noise_ ,n1_b,n_noise_b,n3_b  = singal_prep2( n_/1000, fs,lowcut,lowcut2,highcut,highcut2,order, buffer_time = 200000, resample = True)  
        p1_,p3_,p_noise_ ,p1_b,p_noise_b,p3_b  = singal_prep2( p_, fs,lowcut,lowcut2,highcut,highcut2,order, buffer_time = 200000, resample = False)  
        p4_,n4_,p_peaks2,n_peaks2,noise_p4,noise_n4= peaks_matcher2(p3_b,n3_b,p_noise_b,n_noise_b,arrythmia=arrythmia)
    
        
        for patch, nea, patch_noise,nea_noise, ab in ([p3_,n3_,p_noise_,n_noise_,'a'],[p3_b,n3_b,p_noise_b,n_noise_b,'b']):
                print(name, ab)
                p4_,n4_,p_peaks2,n_peaks2,noise_p4,noise_n4= peaks_matcher2(patch,nea,patch_noise,nea_noise,arrythmia=arrythmia)
                print(len(p_peaks2),len(n_peaks2))
                print(len(p4_),len(n4_))
                p_choice= p_peaks2 if len (p_peaks2)< len(n_peaks2) else n_peaks2
                for counter_each, t in enumerate(range(min( len (p_peaks2),len(n_peaks2))-2)):
                    location = name+'_'+str(counter_each)
                    tp = p_peaks2[t]
                    tn = n_peaks2[t]
                    if tp >1000 and tn > 1000:
                            window_patch_ = p4_[tp-1000:tp+1000]
                            window_nea_   = n4_[tn-1000:tn+1000]
                            ds = ds_finder_iap_patch (window_nea_,window_patch_)
                            window_patch = p4_[tp-ds-1000:tp-ds+7000]
                            window_nea   = n4_[tn-1000:tn+7000]
                            window_noise_patch = noise_p4[tp-ds-1000:tp-ds+7000]
                            window_noise_nea   = noise_n4[tn-1000:tn+7000]     
                            df.loc[location,'name' ]=name
                            df.loc[location,'ch' ]=ab
                            df.loc[location,'#'] = counter_each
                            df.loc[location,'p_span_'+ab]= np.max(window_patch)-np.min(window_patch)
                            df.loc[location,'p_amp_'+ab]= np.max(window_patch)
                            df.loc[location,'p_noise_std_'+ab ]= np.std(window_noise_patch)          
                            df.loc[location,'n_span_'+ab ]= np.max(window_nea)-np.min(window_nea)
                            df.loc[location,'n_amp_'+ab]= np.max(window_nea)
                            df.loc[location,'n_noise_std_'+ab]= np.std(window_noise_nea)  
                            
                            apds_patch_, l_patch = get_apd(window_patch)
                            apds_nea_, l_nea= get_apd(window_nea)
                            l  = round(min(np.min(l_patch),np.min(l_nea)))
                            len_ = round(max(apds_patch_[-1],apds_nea_[-1]))

                            apds_patch = ['APD_patch'+ab+str(i) for i in range(1,11)]
                            apds_nea= ['apds_nea'+ab+str(i) for i in range(1,11)]
                            apds_err = ['APD_Error'+ab+str(i) for i in range(1,11)]
                            sec_peak = np.argmax(window_patch[3000:])+3000
                            sec_min = np.argmin(window_patch[1000:])+1000
                            if sec_peak > sec_min:
                                
                                apds_patch_, l_patch = get_apd(window_patch[:5500])
                                apds_nea_, l_nea= get_apd(window_nea[:5500])
                                l  = round(min(np.min(l_patch),np.min(l_nea)))
                                len_ = round(max(apds_patch_[-1],apds_nea_[-1]))
                            else:
                                apds_patch_, l_patch = get_apd(window_patch)
                                apds_nea_, l_nea= get_apd(window_nea)
                                l  = round(min(np.min(l_patch),np.min(l_nea)))
                                len_ = round(max(apds_patch_[-1],apds_nea_[-1]))
                                
                            df.loc[location,apds_patch] =apds_patch_
                            df.loc[location,apds_nea] =apds_nea_
                            df.loc[location,apds_err] =apds_patch_ -apds_nea_
                       
                                 
                            window_patch_norm = np.array(function1_min_0(window_patch))
                            window_nea_norm   = np.array(function1_min_0(window_nea))
                            df.loc[location,'MAE'+ab] = np.abs( window_patch_norm- window_nea_norm).mean()
                            df.loc[location,'AE_apd'+ab] = np.abs((window_patch_norm[l:l+len_]- window_nea_norm[l:l+len_])).sum()
                            df.loc[location,'R2'+ab] = average_correlation(window_patch,window_nea)
                            patch_sig_pow,patch_noise_pow,patch_sn_power = singal_noise_power(window_patch,window_noise_patch)
                            nea_sig_pow,nea_noise_pow,nea_sn_power = singal_noise_power(np.array(window_nea),np.array(window_noise_nea))  
                            df.loc[location,'patch_sig_pow'+ab] = patch_sig_pow
                            df.loc[location,'patch_noise_pow'+ab] = patch_noise_pow
                            df.loc[location,'patch_sn_power'+ab] = patch_sn_power
                            df.loc[location,'patch_sn_ratio'+ab] = (np.max(window_patch)-np.min(window_patch))/np.std(window_noise_patch)    
                            df.loc[location,'nea_sig_pow'+ab] = nea_sig_pow
                            df.loc[location,'nea_noise_pow'+ab] = nea_noise_pow
                            df.loc[location,'nea_sn_power'+ab] = nea_sn_power
                            df.loc[location,'nea_sn_ratio'+ab] = (np.max(window_nea)-np.min(window_nea))/np.std(window_nea)
                            df.loc[location,'total_counter'+ab] =total_counter
                            df.loc[location,'t_p'+ab] =tp
                            df.loc[location,'t_n'+ab] =tn
                            df.loc[location,'dt_p'+ab] =p_peaks2[t+1]-p_peaks2[t]
                            df.loc[location,'dt_n'+ab] =n_peaks2[t+1]-n_peaks2[t]
                            df.loc[location,'dt_np'+ab] =abs((n_peaks2[t+1]-n_peaks2[t])-(p_peaks2[t+1]-p_peaks2[t]))
                            
                            df.loc[location,'duration'] =p_choice[-1]-p_choice[0]
                            dic_patch_nea[location,ab,'p'] = window_patch
                            dic_patch_nea[location,ab,'n'] = window_nea
                            total_counter = total_counter+1
    #post processing
    
    df2 = df.copy().dropna()
    # discarding the first 40 signals
    df2 = df2[df2['#']>40]
    apds_err_a = ['APD_Error'+'a'+str(i) for i in range(1,11)]
    apds_err_b = ['APD_Error'+'b'+str(i) for i in range(1,11)]
    apds_perr_a = ['APD_PError'+'a'+str(i) for i in range(1,11)]
    apds_perr_b = ['APD_PError'+'b'+str(i) for i in range(1,11)]
    apds_patch_a = ['APD_patch'+'a'+str(i) for i in range(1,11)]
    apds_patch_b = ['APD_patch'+'b'+str(i) for i in range(1,11)]
    df2[apds_perr_a]=np.abs(np.array(df2[apds_err_a]))*100/np.array(df2[apds_patch])
    df2[apds_perr_b]=np.abs(np.array(df2[apds_err_b]))*100/np.array(df2[apds_patch])
    df2[apds_perr_a+apds_perr_b]

    # make the based on second
    df2[['t_pa','t_na','dt_pa', 'dt_na', 'dt_npa']] = df2[['t_pa','t_na','dt_pa', 'dt_na', 'dt_npa']] /10000
    df2[['t_pb','t_nb','dt_pb', 'dt_nb', 'dt_npb']] = df2[['t_pb','t_nb','dt_pb', 'dt_nb', 'dt_npb']] /10000
    df2[apds_err_a] = df2[apds_err_a]/10000
    df2[apds_err_b] = df2[apds_err_b]/10000
                    
    df2['np_max'] =df2.apply(lambda row: max(row['nea_noise_powa'], row['patch_noise_powa']), axis=1)
    df2['sp_min'] =df2.apply(lambda row: min(row['nea_sig_powa'], row['patch_sig_powa']), axis=1)
    df2['sn_min'] =df2.apply(lambda row: min(row['nea_sn_powera'], row['patch_sn_powera']), axis=1)
    df2['sn_max'] =df2.apply(lambda row: max(row['nea_sn_powera'], row['patch_sn_powera']), axis=1)
    df2['amp_min'] =df2.apply(lambda row: min(row['n_amp_a'], row['p_amp_a']), axis=1)
    df2['amp_max'] =df2.apply(lambda row: max(row['n_amp_a'], row['p_amp_a']), axis=1)
    df2['d_sn'] = df2['sn_max']-df2['sn_min']
    df2['d_amp'] = df2['amp_max']-df2['amp_min']

    df3 = df2.reset_index().set_index(['name'])

    return dic_patch_nea, df3
                                                

            
def plot_average_box(df, columns_of_interest):
    fig, axs = plt.subplots(nrows=1, ncols=len(columns_of_interest), figsize=(10, 2), dpi=250)

    # Custom color for the median line and ticks
    median_color = '#FF5555'  # Light Red

    # Adjust space between the plots
    plt.subplots_adjust(wspace=0.2)

    for i, col in enumerate(columns_of_interest):
        ax = axs[i]

        # Group by index and calculate mean for each group
        group_means = df.groupby(df.index).apply(lambda group: group[col].mean())

        # Jittered scatter with very small, semi-transparent dots
        jitter = 0.15 * (np.random.rand(len(group_means)) - 0.5)
        ax.scatter(np.ones(len(group_means)) + jitter, group_means, alpha=0.8, color='black', s=5)

        # Create minimalist box plot
        bp = ax.boxplot(group_means, vert=True, patch_artist=True, showfliers=False, widths=0.4)
        
        # Hide all elements except median line
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], visible=False)
            
        # Style median line
        plt.setp(bp['medians'], color=median_color, linewidth=1.2)

        # Set x-tick labels
        ax.set_xticklabels([col])

        # Hide all spines
        # 'left', 'bottom'
        for spine in ['right', 'top', ]:
            ax.spines[spine].set_visible(False)

        # Remove y-axis labels but keep the ticks, set tick color to subtle grey
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', colors='black')
        ax.tick_params(axis='x', which='both', labelsize=10, length=0)

    # Save the figure
    fig.savefig('patch_all_errorrs.svg', format='svg')

    plt.show()
