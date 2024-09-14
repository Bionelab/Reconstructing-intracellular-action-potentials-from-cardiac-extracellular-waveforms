import sys
import os
import h5py
from multichannels_constants import *
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
from scipy.signal import find_peaks, peak_widths
import matplotlib.colors as mcolors
from scipy.stats import ttest_1samp
from pia_unet_quantile import *
import math
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'EDA_XGBoost'))
from xg_utils import *
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'general'))
from utils import*

def singal_noise_power(spike, noise):

    sp_ = np.mean(spike ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(noise ** 2)  # Square of the signal's amplitude
    sn_ = 20 * math.log10(sp_ / npower_)
    return sp_,npower_,sn_


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



def function1(intra):
  span=max(intra) - intra[0]
  return (intra-(intra)[0])/span


def smoother(preds_,lens1= lens1,pol1 = pol1,lens2 = lens2,pol2 = pol2):
        preds = preds_.reshape(-1,8000).copy()
        preds[:,:1050] = savgol_filter(preds_[:,:1050], lens1, pol1,axis=1)
        preds[:,1000:] = savgol_filter(preds_[:,1000:], lens2, pol2,axis=1)
        return preds


def smoother_single(pred, lens1, pol1, lens2, pol2):
    # Apply Savitzky-Golay filter to the two parts of the signal
    pred[:1050] = savgol_filter(pred[:1050], lens1, pol1)
    pred[900:] = savgol_filter(pred[900:], lens2, pol2)
    return pred
def smoother_process(pred, lens1, pol1, lens2, pol2):
    return smoother_single(pred, lens1, pol1, lens2, pol2)



def smoother_multiprocessing(preds_, lens1= lens1,pol1 = pol1,lens2 = lens2,pol2 = pol2):
    # Reshape preds_ to be a 2D array (n_samples, 8000)
    # Reshape preds_ to be a 2D array (n_samples, 8000)
    preds = preds_.reshape(-1, 8000).copy()

    # Use multiprocessing to parallelize the smoothing of each row
    with ProcessPoolExecutor() as executor:
        preds_smoothed = list(executor.map(
            smoother_process, preds, [lens1]*len(preds), [pol1]*len(preds), [lens2]*len(preds), [pol2]*len(preds)
        ))

    return np.array(preds_smoothed)


def get_all_apds_multiprocessing(all_intras):
    # Using multiprocessing for parallel computation
    with ProcessPoolExecutor() as executor:
        apd_results = list(executor.map(partial(get_apds_single_trace, two_peak_cutoff=round(len(all_intras[0]) * 0.7)), all_intras))
    APD_widths = np.array([w for w, h, l in apd_results])
    return APD_widths



def predictor_all(model1,list_of_data):
    from tensorflow.keras import backend as K
    K.clear_session()
    preds = model1.predict(list_of_data, batch_size=8)
    pred1=preds[0].reshape(-1,8000)
    pred2=preds[1].reshape(-1,8000)
    pred3=preds[2].reshape(-1,8000)
    return pred1,pred2,pred3





def moving_average_smoothing(df, columns, window=5):
    """Apply a moving average to smooth the APD columns."""
    smoothed_df = df.copy()
    for column in columns:
        smoothed_df[column] = smoothed_df[column].rolling(window=window, center=True).mean()
    return smoothed_df.dropna()

def remove_outliers(df, columns, threshold=4):
    """Remove outliers based on z-score across the APD columns."""
    df_cleaned = df.copy()
    for column in columns:
        # Calculate z-scores
        z_scores = np.abs((df_cleaned[column] - df_cleaned[column].mean()) / df_cleaned[column].std())
        # Remove rows where any of the APD columns exceed the threshold
        df_cleaned = df_cleaned[z_scores < threshold]
    return df_cleaned

def return_apds_singlechannel(df, dic_eaps):
    preds1 = np.array([dic_eaps[(i, 'iap_raw_norm')] for i in df.index])
    print(preds1.shape)
    preds1 = preds1[:,0,:].reshape(-1, 8000)
    print(preds1.shape)
    # Using multiprocessing to process all traces in parallel
    apd1 = get_all_apds_multiprocessing(preds1)
    print('ap1 done')
    apd_col1 = ['APD_P1_' + str(i) for i in range(1, 11)]
    print(apd1.shape)
    df2 = df.copy()
    df2.loc[df2.index, apd_col1] = apd1
    df2['p2']=df2['p']/(60*5000)

    return apd1,  df2

def return_apds_3channels(df, dic_eaps):
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

    # Apply moving average smoothing
    df2 = moving_average_smoothing(df2, apd_col1 + apd_col2 + apd_col3, window=5)

    # Remove outliers
    df2_filtered = remove_outliers(df2, apd_col1 + apd_col2 + apd_col3, threshold=5)

    # Filter wrong predictions of APD
    for cols in ['APD_P1_9', 'APD_P3_7', 'APD_P1_5']:
        df2_filtered = df2_filtered[df2_filtered[cols] > 2000]
    df2_filtered['p2']=df2_filtered['p']/(60*5000)
    return apd1, apd2, apd3, df2,df2_filtered
    
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
def all_channels_plotters(df3, t, dic_eaps, iap):
    t2 = t * 5000
    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    indicies = df3.reset_index().set_index(['p2', 'ch']).loc[t]['index'].tolist()
    print(indicies)
    x_spike = 950
    for i, ch in enumerate(indicies):
        ax = axes[i]
        if iap:
            try:
                ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p2')].reshape(-1), color='black')
                ax.fill_between(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p1')].reshape(-1), dic_eaps[(ch, 'p3')], color='red', alpha=0.5)
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
            except:
                ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'iap_raw_norm')].reshape(-1), color='black')
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
                
            
        else:
            ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'eap_raw_norm')].reshape(-1), color='black')
            ax.set_ylim([-0.5, 0.5])
            name = 'eap'

        # Remove all x and y tick labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

def function1(intra):
  span=max(intra) - intra[0]
  return (intra-(intra)[0])/span


def snr(i):
    signal = i[-3000:]
    window_length = len(signal)
    poly_order = 3
    smoothed_signal = savgol_filter(signal, window_length, poly_order)
    noise = np.std(signal-smoothed_signal)
    return np.max(i)/noise

def predictor_all(model1,list_of_data):
    preds = model1.predict(list_of_data)
    pred1=preds[0].reshape(-1,8000)
    pred2=preds[1].reshape(-1,8000)
    pred3=preds[2].reshape(-1,8000)
    return pred1,pred2,pred3
    
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
def screen_noise_noiselevel(data_channel_raw, lowcut,highcut, fs,order):
        
        data_channel = butter_bandpass_filter(data_channel_raw, lowcut,highcut, fs,order=order)
        nois_channel = butter_bandpass_filter(data_channel_raw,highcut+0.001,fs/2.-1,fs)
        noise_level = np.std(butter_bandpass_filter(data_channel,0.8*fs/2.,fs/2.-1,fs))
        return data_channel,nois_channel,noise_level

# finding the peaks for the best channel first
def baseline(data,i , t, fs,lowcut,order,highcut,ps, ds):
    f_data = butter_bandpass_filter(data[i][t[0]:t[1]], lowcut, highcut, fs,order=order) # filtered singal
    f_data = f_data - np.mean(f_data)
    peaks, __ = find_peaks(f_data, prominence=ps, distance=ds) # for extras
    plt.figure(figsize = (40,2))
    plt.plot(f_data)
    plt.plot(peaks,f_data[peaks],'ro')
    plt.show()
    return peaks

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

def remove_outliers_activation(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define a threshold to remove outliers (1.5 * IQR is common)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]



def get_window_max(data_channel_filtered,p,n_points= 850):
            p_data = np.argmax(data_channel_filtered[round(p)-n_points:round(p)+(8000-n_points)]) + round(p)-n_points
            return p_data


def raw_to_seg(data_extras,times,fs,lowcut,highcut,ps, ds,order,baseline_ch, eap_channels, iap_channels ):
    dic_eaps = {}
    main_counter = 0
    df = pd.DataFrame()
    for t_couner, t in enumerate(times[1:]):
        print('t_couner',t_couner)
        #baseline here is used to detect peaks precsiley based on the channel which has the highest amplitude
        b0 = baseline(data_extras,0 ,  t, fs,lowcut,order,highcut,ps, ds )
        for i in range(len(data_extras)):
            if i  in eap_channels:
                print('channel',i,t)
                data_channel_raw = data_extras[i] [t[0]:t[1]] # time segment 
                data_channel_a,nois_channel_a,noise_level = screen_noise_noiselevel(data_channel_raw, lowcut,highcut, fs,order)
                for counter, p in enumerate(b0[2:-2]) :
                    n_points = 950
                    extra_segmented,extra_segmented_a,extra_noise_segmented_a = get_window_with_noise(data_channel_a,data_channel_raw,nois_channel_a,p,n_points= n_points)
                    sp_0,npower_0,sn_0 ,width_extras_a = get_info(extra_segmented,extra_noise_segmented_a,noise_level)
                    std_,span0,snr_=  sn_ratio (extra_segmented)
                    width_extras_raw = get_width(extra_segmented)
                    
                    df.loc[main_counter,['sp','np','sn_p','snr','span','ch','t','p','#','width_extras_a','width_extras_raw','amp','std_',]] = sp_0,npower_0,sn_0,snr_,span0, int(i), int(t_couner),int(p+t[0]),int(counter),width_extras_a,width_extras_raw,np.max(extra_segmented_a)-np.mean(extra_segmented_a[0:100]),std_
                    eap = extra_segmented
                    df.loc[main_counter,'s/n'] = snr(extra_segmented)
                    df.loc[main_counter,'eAP'] = True
                    eap_norm = np.array(function2(extra_segmented)).reshape(-1,8000)
                    eap_norm = eap_norm- eap_norm[:, 0][:, np.newaxis]
                    dic_eaps[(main_counter,'eap_raw')] = extra_segmented
                    dic_eaps[(main_counter,'eap_raw_norm')] = eap_norm
                    dic_eaps[(main_counter,'eap_a_norm')] = function2(extra_segmented_a)
                    
                    p_data = get_window_max(data_channel_a,p,n_points= n_points)              
                    df.loc[main_counter,['p_actual']] =p_data # actual spike time
                    main_counter = main_counter+1
            elif i in iap_channels:
                print('channel',i,t)
                data_channel_raw = data_extras[i] [t[0]:t[1]] # time segment 
                data_channel_a,nois_channel_a,noise_level = screen_noise_noiselevel(data_channel_raw, lowcut,highcut, fs,order)
        
                for counter, p in enumerate(b0[2:-2]) :
                    n_points = 950
                    extra_segmented,extra_segmented_a,extra_noise_segmented_a = get_window_with_noise(data_channel_a,data_channel_raw,nois_channel_a,p,n_points= n_points)
                    sp_0,npower_0,sn_0 ,width_extras_a = get_info(extra_segmented,extra_noise_segmented_a,noise_level)
                    std_,span0,snr_=  sn_ratio (extra_segmented)
                    width_extras_raw = get_width(extra_segmented)
                   
                    
                    df.loc[main_counter,['sp','np','sn_p','snr','span','ch','t','p','#','width_extras_a','width_extras_raw','amp','std_',]] = sp_0,npower_0,sn_0,snr_,span0, int(i), int(t_couner),int(p+t[0]),int(counter),width_extras_a,width_extras_raw,np.max(extra_segmented_a)-np.mean(extra_segmented_a[0:100]),std_
                    df.loc[main_counter,'eAP'] = False
                    eap = extra_segmented
                    df.loc[main_counter,'s/n'] = snr(extra_segmented)
                    eap_norm = np.array(function1(extra_segmented)).reshape(-1)
                    eap_norm = (eap_norm - eap_norm[0]) + 0.11
                    eap_norm = eap_norm/ 1.1
                    eap_norm = eap_norm.reshape(-1,8000)
                    dic_eaps[(main_counter,'iap_raw')] = extra_segmented
                    dic_eaps[(main_counter,'iap_raw_norm')] = eap_norm
                    p_data = get_window_max(data_channel_a,p,n_points= n_points)              
                    df.loc[main_counter,['p_actual']] =p_data # actual spike time
                    main_counter = main_counter+1
            else:
                print('none')

    eaps = np.array([dic_eaps[(i,'eap_raw_norm')] for i in df[df['eAP']==True].index])
    iaps = np.array([dic_eaps[(i,'iap_raw_norm')] for i in df[df['eAP']==False].index])
    eaps = eaps[:,0,:]
    iaps = iaps[:,0,:]
    return dic_eaps,df,eaps,iaps


def all_channels_plotters(df3, t, dic_eaps, iap):
    t2 = t * 5000
    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    
    indicies = df3.reset_index().set_index(['p2', 'ch']).loc[t]['index'].tolist()
    print(indicies)
    
    x_spike = 950
    num_plots = len(indicies)
    
    for i, ch in enumerate(indicies):
        ax = axes[i]
        if iap:
            try:
                ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p2')].reshape(-1), color='black')
                ax.fill_between(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p1')].reshape(-1), dic_eaps[(ch, 'p3')], color='red', alpha=0.5)
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
            except:
                ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'iap_raw_norm')].reshape(-1), color='black')
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
                
        else:
            ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'eap_raw_norm')].reshape(-1), color='black')
            ax.set_ylim([-0.5, 0.5])
            name = 'eap'
        
        # Add channel name text inside each subplot
        ax.text(0.05, 0.9, f'Channel {ch}', transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left', color='blue')

        # Remove all x and y tick labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide the extra blank subplots by setting them invisible
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()



def all_channels_apd_plotters(df3):
    colors = ['#F68B1F', '#6DC8BF', '#92278F']
    darker_colors = [darken_color(c) for c in colors]

    med = 'APD_P2_' + str(5)
    high = 'APD_P3_' + str(5)
    low = 'APD_P1_' + str(5)

    med2 = 'APD_P2_' + str(7)
    high2 = 'APD_P3_' + str(7)
    low2 = 'APD_P1_' + str(7)

    med3 = 'APD_P2_' + str(9)
    high3 = 'APD_P3_' + str(9)
    low3 = 'APD_P1_' + str(9)

    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    df3['p2'] = df3['p'] / (5000 * 60)

    # Calculate global y-limits
    global_min = np.inf
    global_max = -np.inf

    for ch in df3['ch'].unique():
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            apd_values = np.concatenate([
                df_ch[low].values / 5000, df_ch[med].values / 5000, df_ch[high].values / 5000,
                df_ch[low2].values / 5000, df_ch[med2].values / 5000, df_ch[high2].values / 5000,
                df_ch[low3].values / 5000, df_ch[med3].values / 5000, df_ch[high3].values / 5000
            ])
            global_min = min(global_min, np.min(apd_values))
            global_max = max(global_max, np.max(apd_values))

    for i, ch in enumerate(df3['ch'].unique()):
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            yerr = [df_ch[med]/5000 - df_ch[low]/5000, df_ch[high]/5000 - df_ch[med]/5000]
            yerr2 = [df_ch[med2]/5000 - df_ch[low2]/5000, df_ch[high2]/5000 - df_ch[med2]/5000]
            yerr3 = [df_ch[med3]/5000 - df_ch[low3]/5000, df_ch[high3]/5000 - df_ch[med3]/5000]
            axes[i].errorbar(df_ch['p2'], df_ch[med]/5000, yerr=yerr, fmt='o', color=colors[0], ecolor=darker_colors[0], markersize=0.5, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med2]/5000, yerr=yerr2, fmt='o', color=colors[1], ecolor=darker_colors[1], markersize=0.5, alpha=0.5)
            axes[i].errorbar(df_ch['p2'], df_ch[med3]/5000, yerr=yerr3, fmt='o', color=colors[2], ecolor=darker_colors[2], markersize=0.5, alpha=0.5)

            # Apply global y-limits to each subplot
            axes[i].set_ylim(global_min, global_max)

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


def moving_filter(arr, window_size):
    # Simple moving average filter
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')



def return_preds (model,eaps, dic_eaps,df_):
    df = df_.copy()
    dic_eaps2 = dic_eaps.copy()
    pred1,pred2,pred3 =  predictor_all(model,eaps)
    preds1_sm = smoother_multiprocessing(pred1)
    preds2_sm = smoother_multiprocessing(pred2)
    preds3_sm = smoother_multiprocessing(pred3)

    for p1,p2,p3,ps1,ps2,ps3, ind in zip (pred1,pred2,pred3,
                                             preds1_sm,preds2_sm,preds3_sm,
                                          df[df['eAP']==True].index):
            dic_eaps2[(ind,'p1')]=p1
            dic_eaps2[(ind,'p2')]=p2
            dic_eaps2[(ind,'p3')]=p3
            dic_eaps2[(ind,'p1_smooth')]=ps1
            dic_eaps2[(ind,'p2_smooth')]=ps2
            dic_eaps2[(ind,'p3_smooth')]=ps3
    return dic_eaps2,df

##
def single_channel_apd_plotters(df3):
    colors = ['#F68B1F', '#6DC8BF', '#92278F']
    # darker_colors = [darken_color(c) for c in colors]
    darker_color = colors
    med = 'APD_P1_' + str(5)
    med2 = 'APD_P1_' + str(7)
    med3 = 'APD_P1_' + str(9)

    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    df3['p2'] = df3['p'] / (5000 * 60)

    for i, ch in enumerate(df3['ch'].unique()):
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            # Plot lines instead of dots, using dashed style
            axes[i].plot(df_ch['p2'], df_ch[med]/5000,  color=colors[0], alpha=0.5)
            axes[i].plot(df_ch['p2'], df_ch[med2]/5000,  color=colors[1], alpha=0.5)
            axes[i].plot(df_ch['p2'], df_ch[med3]/5000,  color=colors[2], alpha=0.5)

            # Apply fixed y-limits to each subplot (between 0.4 and 1.2)
            axes[i].set_ylim(0.4, 1.2)

            # Show y-axis for leftmost plots, hide for the rest
            if i % 17 == 0:  # Leftmost column (first of every 17 subplots)
                axes[i].tick_params(axis='y', labelsize=8)  # Show y-axis ticks and labels
            # else:
                # axes[i].set_yticklabels([])
                # axes[i].set_yticks([])

            # Remove x-axis ticks and labels
            # axes[i].set_xticklabels([])
            # axes[i].set_xticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # Save as SVG to ensure it's not rasterized
    # plt.savefig('results/apd_splot.svg', format='svg')
    plt.show()


def all_channels_apd_plotters(df3):
    colors = ['#F68B1F', '#6DC8BF', '#92278F']
    darker_colors = [darken_color(c) for c in colors]

    med = 'APD_P2_' + str(5)
    high = 'APD_P3_' + str(5)
    low = 'APD_P1_' + str(5)

    med2 = 'APD_P2_' + str(7)
    high2 = 'APD_P3_' + str(7)
    low2 = 'APD_P1_' + str(7)

    med3 = 'APD_P2_' + str(9)
    high3 = 'APD_P3_' + str(9)
    low3 = 'APD_P1_' + str(9)

    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    df3['p2'] = df3['p'] / (5000 * 60)

    # Calculate global y-limits
    global_min = np.inf
    global_max = -np.inf

    for ch in df3['ch'].unique():
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            apd_values = np.concatenate([
                df_ch[low].values / 5000, df_ch[med].values / 5000, df_ch[high].values / 5000,
                df_ch[low2].values / 5000, df_ch[med2].values / 5000, df_ch[high2].values / 5000,
                df_ch[low3].values / 5000, df_ch[med3].values / 5000, df_ch[high3].values / 5000
            ])
            global_min = min(global_min, np.min(apd_values))
            global_max = max(global_max, np.max(apd_values))
        else:
            print(ch)

    for i, ch in enumerate(df3['ch'].unique()):
        df_ch = df3[df3['ch'] == ch]
        if len(df_ch) > 10:
            yerr = [df_ch[med]/5000 - df_ch[low]/5000, df_ch[high]/5000 - df_ch[med]/5000]
            yerr2 = [df_ch[med2]/5000 - df_ch[low2]/5000, df_ch[high2]/5000 - df_ch[med2]/5000]
            yerr3 = [df_ch[med3]/5000 - df_ch[low3]/5000, df_ch[high3]/5000 - df_ch[med3]/5000]
            
            # Rasterized plot to reduce file size
            axes[i].errorbar(df_ch['p2'], df_ch[med]/5000, yerr=yerr, fmt='o', color=colors[0], ecolor=darker_colors[0], markersize=1, alpha=0.5, rasterized=True)
            axes[i].errorbar(df_ch['p2'], df_ch[med2]/5000, yerr=yerr2, fmt='o', color=colors[1], ecolor=darker_colors[1], markersize=1, alpha=0.5, rasterized=True)
            axes[i].errorbar(df_ch['p2'], df_ch[med3]/5000, yerr=yerr3, fmt='o', color=colors[2], ecolor=darker_colors[2], markersize=1, alpha=0.5, rasterized=True)

            # Apply fixed y-limits to each subplot (between 0.4 and 1.2)
            axes[i].set_ylim(0.4, 1.2)

            # Show y-axis for leftmost plots, hide for the rest
            if i % 17 == 0:  # Leftmost column (first of every 17 subplots)
                axes[i].tick_params(axis='y', labelsize=8)  # Show y-axis ticks and labels
            else:
                axes[i].set_yticklabels([])
                axes[i].set_yticks([])

            # Remove x-axis ticks and labels
            axes[i].set_xticklabels([])
            axes[i].set_xticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # Save as SVG and ensure it's rasterized
    # plt.savefig('results/apd_splot2.svg', format='svg')
    plt.show()


def all_channels_plotters(df3, t, dic_eaps, iap, raster = True, save_path=None):
    t2 = t * 5000
    fig, axes = plt.subplots(4, 17, figsize=(34, 8), dpi=200)
    axes = axes.flatten()
    
    indicies = df3.reset_index().set_index(['p2', 'ch']).loc[t]['index'].tolist()
    print(indicies)
    
    x_spike = 950
    num_plots = len(indicies)
    
    for i, ch in enumerate(indicies):
        ax = axes[i]
        channel = df3.loc[ch,'ch']
        if iap:
            
            try:
               
                line1, = ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p2')].reshape(-1), color='black')
                fill1 = ax.fill_between(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'p1')].reshape(-1), dic_eaps[(ch, 'p3')], color='red', alpha=0.5)
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
            except:
                line1, = ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'iap_raw_norm')].reshape(-1), color='black')
                ax.set_ylim([-0.2, 1.2])
                name = 'iap'
                
        else:
            line1, = ax.plot(np.arange(t2 - x_spike, t2 + 8000 - x_spike) / (5000 * 60), dic_eaps[(ch, 'eap_raw_norm')].reshape(-1), color='black')
            ax.set_ylim([-0.5, 0.5])
            name = 'eap'
        
        # Add channel name text inside each subplot
        # ax.text(0.05, 0.9, f'{channel}', transform=ax.transAxes, fontsize=8,
        #         verticalalignment='top', horizontalalignment='left', color='blue')

        # Remove all x and y tick labels and ticks
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticks([])
        # ax.set_yticks([])

        # Rasterize the content inside the current plot
        if raster:
            line1.set_rasterized(True)
            if 'fill1' in locals():
                fill1.set_rasterized(True)

    # Hide the extra blank subplots by setting them invisible
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()

    # Save the figure as an SVG with rasterized elements
    # if save_path:
        # plt.savefig(save_path, format='svg', dpi=300, rasterized=raster)
    
    plt.show()



def darken_color(color, factor=0.6):
    # Convert hex color to RGB
    rgb = mcolors.hex2color(color)
    # Darken the RGB color
    dark_rgb = [x * factor for x in rgb]
    return mcolors.to_hex(dark_rgb)

def apd_changes_summary(df_, drug_time_in_sec, perc=False):
    df = df_.copy()
    df = df.set_index('ch')
    df['p2']=df['p']/(60*5000)
    pre_drug = df[df['p2']<(drug_time_in_sec/60)]
    pro_drug = df[df['p2']>(drug_time_in_sec/60)]
    pre_drug
    apd_data= pd.DataFrame()
    counter= 0
    for i in [3,5,7,9]:
        for ch in df_['ch'].unique():
    
            ch = int(ch)
            try:
                pro_drug_df= pro_drug.loc[ch].sort_values('p2')
                pre_drug_df= pre_drug.loc[ch].sort_values('p2')
                apd_pre = pre_drug_df.iloc[-10:]['APD_P1_'+str(i)].median()
            except:
                apd_pre = pro_drug_df.iloc[:5]['APD_P1_'+str(i)].median()
                
            apd_pro = pro_drug_df.sort_values('p2').iloc[-20:]['APD_P1_'+str(i)].median()
            apd_data.loc[counter,'mean'] = apd_pro - apd_pre
            apd_data.loc[counter,'mean_perc'] = (apd_pro - apd_pre)*100/apd_pre
            apd_data.loc[counter,'apd'] = int(i)
            apd_data.loc[counter,'ch'] = int(ch)
            counter = counter + 1

    display(apd_data)
    # Define colors
    box_colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, silver, bronze
    dot_color = 'black'  # Color for the individual data points
    
    # Create a boxplot
    plt.figure(figsize=(4, 4), dpi=250)
    
    # Significance levels
    significance_markers = []  # List to hold significance markers for each group
    
    # Create box plots for each APD group
    for i, apd in enumerate(apd_data['apd'].unique()):
        # Filter the DataFrame for the current APD group
        if perc == False:
            apd_data_ = apd_data[apd_data['apd'] == apd]['mean'] / 5
            y_label = 'Drug Induced APD Change (ms)'
            name = 'ms'
        else:
            apd_data_ = apd_data[apd_data['apd'] == apd]['mean_perc']
            y_label = 'Drug Induced APD Change %'
            name = 'perc'
        
        # Add boxplot for the APD group
        bp = plt.boxplot(apd_data_, positions=[i], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=box_colors[i % len(box_colors)], color=dot_color),
                         medianprops=dict(color='orange'),
                         whiskerprops=dict(color=dot_color),
                         capprops=dict(color=dot_color),
                         flierprops=dict(markerfacecolor=dot_color, marker='o', markersize=1))
        
        # Add scatter plot for individual data points
        plt.scatter(np.random.normal(i, 0.04, size=len(apd_data_)), apd_data_,
                    color=dot_color, alpha=1, edgecolor=None, zorder=2, s=2)
        
        # Perform a one-sample t-test to test if the mean of apd_data is significantly different from 0
        t_stat, p_value = ttest_1samp(apd_data_, 0)
        
        # Determine significance level and prepare marker
        mean_value = np.mean(apd_data_)
        if p_value < 0.001:
            marker = '***'
        elif p_value < 0.01:
            marker = '**'
        elif p_value < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        
        significance_markers.append((i, mean_value, marker, p_value))
    
    # Customize the plot
    plt.xticks(range(len(apd_data['apd'].unique())), apd_data['apd'].unique())
    plt.ylabel(y_label, fontsize=14)
    
    # Add significance stars and p-values on top of each box plot
    for i, mean_value, marker, p_value in significance_markers:
        plt.text(i, max(apd_data_) + 0.05 * max(apd_data_), marker, ha='center', va='bottom', fontsize=10, color='red')
        plt.text(i, max(apd_data_) + 0.1 * max(apd_data_), f'p={p_value:.3f}', ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    # Save the figure
    # plt.savefig('results/apd_box_plots' + name + '.svg', format='svg', dpi=300)
    
    # Show the plot
    plt.show()
def find_in(df, target):
    differences = np.abs(df['p2'] - target)
    closest_index = df.index[np.argmin(differences)]
    return closest_index

def samples_from_plot_channel_change_over_time(dic_eaps,ch,df3, times_to_plot_in_sec = [200,350,500,650]):
    multi_channels_info_dic = dic_eaps.copy()
    print('ch',ch)
    df_ch = df3[df3['ch']==ch]
    plt.figure(dpi = 100, figsize = (4,3))
    colors = ['#F68B1F','#6DC8BF','black','#92278F']
    colors = ['#F68B1F', '#6DC8BF', '#92278F', '#345995']
    for counter,i in enumerate(times_to_plot_in_sec):
        print(counter)
        ind = find_in(df_ch, i/60)
        print(i/60,ind,df3.loc[ind,'p2'])
        y = multi_channels_info_dic[(ind,'p2')]
        y1 = multi_channels_info_dic[(ind,'p1')]
        y3 = multi_channels_info_dic[(ind,'p3')]
        plt.plot(np.arange(8000)/5000, y.tolist(), 
                 color =colors [counter], label = 'at '+str(round( i))+'(s)')
        plt.fill_between(np.arange(8000)/5000,
                         y1,
                         y3, 
                         color = colors [counter], alpha=0.1)
    
    plt.gca().get_yaxis().set_visible(False)
    
    # # Only show ticks on the left and bottom spines
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc=(0.6, 0.6), fontsize = 12, )
    # plt.savefig("results/4windows_together.svg", format="svg",  dpi = 350)
    plt.show()


def activation_vs_apd(df_raw, df_with_apd, apd_numb = 5):

    df = df_raw.copy()
    df['p_diff'] = None
    # Loop through each unique value of the 'p' column
    for p_value in df['p'].unique():
        # Find the p_data value where ch == 0 and this p_value
        p_data_ch0 = df[(df['ch'] == 0) & (df['p'] == p_value)]['p_actual'].values[0]
        # Subtract this p_data_ch0 from all rows with the same p_value
        df.loc[df['p'] == p_value, 'p_diff'] = df.loc[df['p'] == p_value, 'p_actual'] - p_data_ch0
    
    df_apd = df_with_apd
    
    # Create the new column 'apd_p5_diff' by first setting it to NaN
    df_apd['apd_p'+str(apd_numb)+'_diff'] = None
    
    # Loop through each unique value of the 'p' column
    for p_value in df_apd['p'].unique():
        # Find the p_data value where ch == 0 and this p_value
        p_data_ch0 = df_apd[(df_apd['ch'] == 1) & (df_apd['p'] == p_value)]['APD_P2_5'].values
        
        # Proceed only if p_data_ch0 is not empty
        if len(p_data_ch0) > 0:
            p_data_ch0 = p_data_ch0[0]
            
            # Subtract this p_data_ch0 from all rows with the same p_value
            df_apd.loc[df_apd['p'] == p_value, 'apd_p5_diff'] = df_apd.loc[df_apd['p'] == p_value, 'APD_P2_5'] - p_data_ch0
        else:
            # Handle cases where ch == 0 for the given p_value does not exist
            print(f"No ch == 0 for p_value {p_value}")
    
    # Now df_apd has the new column 'apd_p5_diff'


    # Set larger font sizes for labels and titles
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18})
    # Create the subplots in one row
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    # Remove outliers from the 'p_diff' column and create the first heatmap
    df_clean1 = remove_outliers_activation(df, 'p_diff')
    # Remove outliers from the 'apd_p5_diff' column and create the second heatmap
    df_clean2 = remove_outliers_activation(df_apd, 'apd_p5_diff')
    # Filter df_clean1 to only include 'ch' values that are also in df_clean2
    filtered_df_clean1 = df_clean1[df_clean1['ch'].isin(df_clean2['ch'])]
    # Convert the 'p' values into minutes and round
    filtered_df_clean1['p'] = round(filtered_df_clean1['p'] / (60 * 5000), 2)
    df_clean2['p'] = round(df_clean2['p'] / (60 * 5000), 2)
    # Normalize differences
    filtered_df_clean1['p_diff'] = filtered_df_clean1['p_diff'] / 5
    df_clean2['apd_p5_diff'] = df_clean2['apd_p5_diff'] / 5
    # Pivot the data for heatmap
    heatmap_data1 = filtered_df_clean1.pivot_table(index='ch', columns='p', values='p_diff')
    heatmap_data2 = df_clean2.pivot_table(index='ch', columns='p', values='apd_p5_diff')
    # Plot the first heatmap
    sns.heatmap(heatmap_data1, cmap='viridis', cbar=True, ax=axes[0])
    axes[0].set_title('Activation Map (Spike time difference) [ms]')
    axes[0].set_xlabel('Time [min]')
    axes[0].set_ylabel('Channel #')
    # Plot the second heatmap
    sns.heatmap(heatmap_data2, cmap='viridis', cbar=True, ax=axes[1])
    axes[1].set_title('APD5 Map (APD 5 difference) [ms]')
    axes[1].set_xlabel('Time [min]')
    axes[1].set_ylabel('')
    # Ensure consistent y-axis labels and ticks
    axes[1].set_yticks(axes[0].get_yticks())
    axes[1].set_yticklabels(axes[0].get_yticklabels())
    # Adjust layout
    plt.tight_layout()
    # Show the heatmaps
    plt.show()





def function1_min_0(intra):
    min_val = min(intra[1000:])
    span = max(intra) - min_val
    return [(val - min_val) / span for val in intra]


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = 2 * (array - min_value) / (max_value - min_value) - 1
    return normalized_array



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


def name_from_dic  (dic, data_num):
    file_name = 'data-'+str(data_num)+'__'
    for name,val in dic.items():
        file_name = file_name + name+'-'+str(val)+'__'
    file_name = file_name+'.h5'
    return file_name


