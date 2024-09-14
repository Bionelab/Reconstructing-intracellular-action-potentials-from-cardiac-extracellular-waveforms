import os
import sys
from constants import *
general_utils_dir = os.path.join(os.path.dirname(os.getcwd()), 'general')
sys.path.append(general_utils_dir)
from  general_utils import *  # replace 'your_module' with the name of your Python file without the .py extension
sys.path.remove(general_utils_dir)



def singal_noise_power(spike, noise):
    sp_ = np.mean(spike ** 2)  # Square of the signal's amplitude
    npower_ = np.mean(noise ** 2)  # Square of the signal's amplitude
    sn_ = 10 * math.log10(sp_ / npower_)

    return sp_,npower_,sn_


# neighboring channles iap
def baseline_seg(data_i , t, fs,lowcut,order,highcut,ps, ds,eap ):
    if eap==True:
        f_data = butter_bandpass_filter(data_i[t[0]:t[1]], lowcut, highcut, fs,order=order) # filtered singal
    else:
        f_data = butter_bandpass_filter(data_i[t[0]:t[1]], lowcut, highcut, fs,order=order) # filtered singal
    
    f_data = f_data - np.mean(f_data)
    peaks, __ = find_peaks(f_data, prominence=ps, distance=ds) # for extras
    plt.figure(figsize = (40,2))
    plt.plot(f_data)
    plt.plot(peaks,f_data[peaks],'ro')
    plt.show()
    return peaks


def eap_window_maker2(fs,lowcut,lowcut2,order,highcut1,highcut2, data,lm_baseline, t ):
        extra1 = data[t[0]:t[1]]/1000
        f_extra1 = butter_bandpass_filter(extra1, lowcut, highcut1, fs,order=order) # filtered singal
        n_extra1 = butter_bandpass_filter(extra1,highcut+0.001,fs/2.-1,fs, order=order)#get high frequency component noise 
        
        f_extra2 = butter_bandpass_filter(extra1, lowcut2, highcut2, fs,order=order) # filtered singal 
        n_extra2 = butter_bandpass_filter(extra1,0.8*(fs/2),fs/2.-1,fs, order=order)#get high frequency component noise 

        lm1 = []
        for p in lm_baseline:
            p_new = np.argmax(f_extra1[p-200:p+200]) + p-200
            lm1.append(p_new)
        fw1s=[]
        fw2s=[]
        nw1s=[]
        nw2s=[]
        rw1s=[]
    
        for n0 in range(1,len(lm1)-1):
            r_w1 = extra1[lm1[n0]-1000:lm1[n0]+7000]
            rw1s.append(r_w1)
            f_w1 = f_extra1[lm1[n0]-1000:lm1[n0]+7000]
            fw1s.append(f_w1)
            f_w2 = f_extra2[lm1[n0]-1000:lm1[n0]+7000]
            fw2s.append(f_w2)
            n_w1 = n_extra1[lm1[n0]-1000:lm1[n0]+7000]
            nw1s.append(n_w1)
            n_w2 = n_extra2[lm1[n0]-1000:lm1[n0]+7000]
            nw2s.append(n_w2)

            if n0 ==1 :
                eap_first = f_w1 -  np.mean(f_w1 [:200])
            if  n0 ==len(lm1)-2:
                eap_last = f_w1- np.mean(f_w1 [:200])
                # ax.plot(normalize_array(f_w1), label = ch_number)
            # plt.show()
            
        return rw1s,fw1s,fw2s,nw1s,nw2s, lm1[1:-1],eap_first,eap_last # raw, only trend filtered, trend and noise filtered, nosie, peaks time or time points

    


def iap_neighboring_comp(combined_dict,times,availablities,baseline_channel):
    total_counter = 0
    dic_of_iaps_a= {}
    dic_of_iaps_b= {}
    dic_of_iaps_r= {}
    df = pd.DataFrame()

    for name, i, j in combined_dict.keys():
        print(name,i,j)
        data= combined_dict[(name, i, j)]
        data_i,data_j  = data[0], data[1]
        t_l = times[name]
        print('time',t_l)
        df_av = availablities[name]
        b_ch = baseline_channel[name]
        for t_ in range(len(t_l)-1):
            t_ = t_+1
            b_aip =  baseline_seg(combined_dict[b_ch][0] , t_l[t_], fs,lowcut_iap,order,highcut_iap,ps_iap/(1.2*t_),ds, eap = False )

            if df_av.loc[i,'t'+str(t_)] == True and df_av.loc[j,'t'+str(t_)] == True:
                    rw1s_i,fw1s_i,fw2s_i,nw1s_i,nw2s_i, ts_i,_,_= eap_window_maker2(fs,lowcut_iap,lowcut_iap2,order,highcut_iap,highcut_iap2,data_i,b_aip, t_l[t_] )
                    rw1s_j,fw1s_j,fw2s_j,nw1s_j,nw2s_j, ts_j,_,_= eap_window_maker2(fs,lowcut_iap,lowcut_iap2,order,highcut_iap,highcut_iap2,data_j,b_aip, t_l[t_] )

                    for spike_counter in range(len(rw1s_i)):

                        df.loc[total_counter,'name'] = name
                        df.loc[total_counter,'ch1'] = i
                        df.loc[total_counter,'ch2'] = j
                        df.loc[total_counter,'t_i'] = ts_i[spike_counter]+t_l[t_] [0]
                        df.loc[total_counter,'t_j'] = ts_j[spike_counter]+t_l[t_] [0]

                        spike1a = fw1s_i[spike_counter]
                        spike2a = fw1s_j[spike_counter]
                        noise1a = nw1s_i[spike_counter]
                        noise2a = nw1s_j[spike_counter]

                        noise1a_std = np.std(noise1a)#get high frequency component
                        noise2a_std = np.std(noise2a)#get high frequency component
                        span_ai = np.max(spike1a)-np.min(spike1a)
                        span_aj = np.max(spike2a)-np.min(spike2a)

                        spike1b = fw2s_i[spike_counter]
                        spike2b = fw2s_j[spike_counter]
                        noise1b = nw2s_i[spike_counter]
                        noise2b = nw2s_j[spike_counter]

                        noise1b_std = np.std(noise1b)#get high frequency component
                        noise2b_std = np.std(noise2b)#get high frequency component
                        span_bi = np.max(spike1b)-np.min(spike1b)
                        span_bj = np.max(spike2b)-np.min(spike2b)
                        spike1r = rw1s_i[spike_counter]
                        spike2r = rw1s_j[spike_counter]
                        apdsi_a, l_i_a = get_apd(spike1a)
                        apdsj_a, l_j_a  = get_apd(spike2a)
                        apdsi_b, l_i_b  = get_apd(spike1b)
                        apdsj_b, l_j_b  = get_apd(spike2b)
                        apdsi_r, l_i_r  = get_apd(spike1r)
                        apdsj_r, l_j_r  = get_apd(spike2r)
                        df.loc[total_counter,'MAE_a'] = np.abs(normalize_array(spike1a)-normalize_array(spike2a)).mean()
                        df.loc[total_counter,'MAE_b'] = np.abs(normalize_array(spike1b)-normalize_array(spike2b)).mean()
                        df.loc[total_counter,'MAE_r'] = np.abs(normalize_array(spike1r)-normalize_array(spike2r)).mean()
                        l  = round(min(np.min(l_i_r),np.min(l_j_r)))
                        len_ = round(max(apdsi_r[-1],apdsj_r[-1]))
                        df.loc[total_counter,'MAE_r_apd'] = np.abs(normalize_array(spike1r[l:l+len_])-normalize_array(spike2r[l:l+len_])).mean()
                        df.loc[total_counter,'AE_r_apd'] = np.abs(normalize_array(spike1r[l:l+len_])-normalize_array(spike2r[l:l+len_])).sum()
                        l  = round(min(np.min(l_i_a),np.min(l_j_a)))
                        len_ = round(max(apdsi_a[-1],apdsj_a[-1]))
                        df.loc[total_counter,'MAE_a_apd'] = np.abs(normalize_array(spike1a[l:l+len_])-normalize_array(spike2a[l:l+len_])).mean()
                        df.loc[total_counter,'AE_a_apd'] = np.abs(normalize_array(spike1a[l:l+len_])-normalize_array(spike2a[l:l+len_])).sum()
                        l  = round(min(np.min(l_i_b),np.min(l_j_b)))
                        len_ = round(max(apdsi_b[-1],apdsj_b[-1]))
                        df.loc[total_counter,'MAE_b_apd'] = np.abs(normalize_array(spike1b[l:l+len_])-normalize_array(spike2b[l:l+len_])).mean()
                        df.loc[total_counter,'AE_b_apd'] = np.abs(normalize_array(spike1b[l:l+len_])-normalize_array(spike2b[l:l+len_])).sum()
                        df.loc[total_counter,'R2_i'] = average_correlation(spike1a,spike2a)
                        df.loc[total_counter,'R2_j'] = average_correlation(spike1b,spike2b)
                        df.loc[total_counter,'R2_r'] = average_correlation(spike1r,spike2r)
                        df.loc[total_counter,apds_err_a] =apdsi_a - apdsj_a
                        df.loc[total_counter,apds_i_a] =apdsi_a 
                        df.loc[total_counter,apds_j_a] =apdsj_a 
                        df.loc[total_counter,apds_err_b] =apdsi_b - apdsj_b
                        df.loc[total_counter,apds_i_b] =apdsi_b 
                        df.loc[total_counter,apds_j_b] =apdsj_b 
                        df.loc[total_counter,apds_err_r] =apdsi_r - apdsj_r
                        df.loc[total_counter,apds_i_r] =apdsi_r 
                        df.loc[total_counter,apds_j_r] =apdsj_r 
                        sp_a,npower_a,sn_a = singal_noise_power(spike1a,noise1a)
                        df.loc[total_counter,'sp_a_i'] = sp_a
                        df.loc[total_counter,'npower_a_i'] = npower_a
                        df.loc[total_counter,'sn_a_i'] = sn_a
                        df.loc[total_counter,'amp_a_i'] = np.max(spike1a)
                        df.loc[total_counter,'span_a_i'] = span_ai
                        df.loc[total_counter,'noise_std_a_i'] = noise1a_std
                        df.loc[total_counter,'sn_a_i_ratio'] = span_ai/noise1a_std
                        sp_a,npower_a,sn_a = singal_noise_power(spike2a,noise2a)
                        df.loc[total_counter,'sp_a_j'] = sp_a
                        df.loc[total_counter,'npower_a_j'] = npower_a
                        df.loc[total_counter,'sn_a_j'] = sn_a
                        df.loc[total_counter,'amp_a_j'] = np.max(spike2a)
                        df.loc[total_counter,'span_a_j'] =span_aj
                        df.loc[total_counter,'noise_std_a_j'] = noise2a_std
                        df.loc[total_counter,'sn_a_j_ratio'] = span_aj/noise2a_std
                        sp_b,npower_b,sn_b = singal_noise_power(spike1b,noise1b)
                        df.loc[total_counter,'sp_b_i'] = sp_b
                        df.loc[total_counter,'npower_b_i'] = npower_b
                        df.loc[total_counter,'sn_b_i'] = sn_b
                        df.loc[total_counter,'amp_b_i'] = np.max(spike1b)
                        df.loc[total_counter,'span_b_i'] =span_bi
                        df.loc[total_counter,'noise_std_b_i'] = noise1b_std
                        df.loc[total_counter,'sn_b_i_ratio'] = span_bi/noise1b_std
                        sp_b,npower_b,sn_b = singal_noise_power(spike2b,noise2b)
                        df.loc[total_counter,'sp_b_j'] = sp_b
                        df.loc[total_counter,'npower_b_j'] = npower_b
                        df.loc[total_counter,'sn_b_j'] = sn_b
                        df.loc[total_counter,'amp_b_j'] = np.max(spike2b)
                        df.loc[total_counter,'span_b_j'] =span_bj
                        df.loc[total_counter,'noise_std_b_j'] = noise2b_std
                        df.loc[total_counter,'sn_b_j_ratio'] = span_bj/noise2b_std
                        dic_of_iaps_a['i',total_counter]=spike1a
                        dic_of_iaps_a['j',total_counter]=spike2a
                        dic_of_iaps_b['i',total_counter]=spike1b
                        dic_of_iaps_b['j',total_counter]=spike2b
                        dic_of_iaps_r['i',total_counter]=spike1r
                        dic_of_iaps_r['j',total_counter]=spike2r
                        total_counter = total_counter + 1



    df.to_csv('results/neighboring_info_v2.csv')     
    with open('neighboring_info_a.pkl', 'wb') as file:
        pickle.dump(dic_of_iaps_a, file)
    with open('results/neighboring_info_b.pkl', 'wb') as file:
        pickle.dump(dic_of_iaps_b, file)
    with open('results/neighboring_info_r.pkl', 'wb') as file:
        pickle.dump(dic_of_iaps_r, file)
    
    return post_process(df), dic_of_iaps_a,dic_of_iaps_b,dic_of_iaps_r


def post_process(df2):
    df = df2.copy()
    df['ch_pairs'] = df.groupby(['ch1', 'ch2','name']).ngroup()
    df['sn_a_max']= df.apply(lambda row: max(row['sn_a_i'], row['sn_a_j']), axis=1)
    df['sn_b_max']= df.apply(lambda row: max(row['sn_b_i'], row['sn_b_j']), axis=1)

    df['sn_ar_max']= df.apply(lambda row: max(row['sn_a_i_ratio'], row['sn_a_j_ratio']), axis=1)
    df['sn_br_max']= df.apply(lambda row: max(row['sn_b_i_ratio'], row['sn_b_j_ratio']), axis=1)

    df['sn_a_diff']= abs(df['sn_a_i']-df['sn_a_j'])
    df['sn_b_diff']= abs(df['sn_b_i']-df['sn_b_j'])
    df['sn_ar_diff']= abs(df['sn_a_i_ratio']-df['sn_a_j_ratio'])
    df['sn_br_diff']= abs(df['sn_b_i_ratio']-df['sn_b_j_ratio'])

    df['amp_a_diff']= abs(df['amp_a_i']-df['amp_a_j'])
    df['amp_b_diff']= abs(df['amp_b_i']-df['amp_b_j'])
    df['amp_a_diff']= abs(df['amp_a_i']-df['amp_a_j'])

    df['sn_a_i_round'] =df['sn_a_i'].round()
    df['sn_a_j_round'] =df['sn_a_j'].round()
    df['sn_b_i_round'] =df['sn_b_i'].round()
    df['sn_b_j_round'] =df['sn_b_j'].round()
    df['sn_ar_i_round'] =df['sn_a_i_ratio'].round()
    df['sn_ar_j_round'] =df['sn_a_j_ratio'].round()
    df['sn_br_i_round'] =df['sn_b_i_ratio'].round()
    df['sn_br_j_round'] =df['sn_b_j_ratio'].round()


    df['amp_a_min']= df.apply(lambda row: min(row['amp_a_i'], row['amp_a_j']), axis=1)
    df['amp_b_min']= df.apply(lambda row: min(row['amp_b_i'], row['amp_b_j']), axis=1)
    df['sn_a_min']= df.apply(lambda row: min(row['sn_a_i'], row['sn_a_j']), axis=1)

    df['sn_a_com']= df.apply(lambda row: row['sn_a_i'] *row['sn_a_j'], axis=1)

    df['sn_b_min']= df.apply(lambda row: min(row['sn_b_i'], row['sn_b_j']), axis=1)
    df['sn_ar_min']= df.apply(lambda row: min(row['sn_a_i_ratio'], row['sn_a_j_ratio']), axis=1)
    df['sn_br_min']= df.apply(lambda row: min(row['sn_b_i_ratio'], row['sn_b_j_ratio']), axis=1)
    df['sn_min_a_round']= df.apply(lambda row: min(row['sn_a_i_round'], row['sn_a_j_round']), axis=1)
    df['sn_max_a_round']= df.apply(lambda row: max(row['sn_a_i_round'], row['sn_a_j_round']), axis=1)
    df['sn_min_b_round']= df.apply(lambda row: min(row['sn_b_i_round'], row['sn_b_j_round']), axis=1)
    df['sn_max_b_round']= df.apply(lambda row: max(row['sn_b_i_round'], row['sn_b_j_round']), axis=1)
    df['sn_diff_a_round']= abs(df['sn_a_i_round']-df['sn_a_j_round'])
    df['sn_diff_b_round']= abs(df['sn_b_i_round']-df['sn_b_j_round'])

    df['sn_min_a_round']= df.apply(lambda row: min(row['sn_ar_i_round'], row['sn_ar_j_round']), axis=1)
    df['sn_max_a_round']= df.apply(lambda row: max(row['sn_ar_i_round'], row['sn_ar_j_round']), axis=1)
    df['sn_min_b_round']= df.apply(lambda row: min(row['sn_br_i_round'], row['sn_br_j_round']), axis=1)
    df['sn_max_b_round']= df.apply(lambda row: max(row['sn_br_i_round'], row['sn_br_j_round']), axis=1)
    df['sn_diff_a_round']= abs(df['sn_ar_i_round']-df['sn_ar_i_round'])
    df['sn_diff_b_round']= abs(df['sn_br_i_round']-df['sn_br_j_round'])


    df['noise_std_b_max']= df.apply(lambda row: max(row['noise_std_b_i'], row['noise_std_b_j']), axis=1)
    df['noise_std_a_max']= df.apply(lambda row: max(row['noise_std_a_i'], row['noise_std_a_j']), axis=1)


    df['noise_std_b_min']= df.apply(lambda row: min(row['noise_std_b_i'], row['noise_std_b_j']), axis=1)
    df['noise_std_a_min']= df.apply(lambda row: min(row['noise_std_a_i'], row['noise_std_a_j']), axis=1)


    df[apds_err_a]= np.abs(df[apds_err_a])
    df[apds_err_b]= np.abs(df[apds_err_b])
    df[apds_err_r]= np.abs(df[apds_err_r])



    df.loc[df['sn_a_max'] == df['sn_a_i'],apds_perr_a ]=np.array(df.loc[df['sn_a_max'] == df['sn_a_i'],apds_err_a])*100/np.array(df.loc[df['sn_a_max'] == df['sn_a_i'],apds_i_a])
    df.loc[df['sn_a_max'] == df['sn_a_j'],apds_perr_a ]=np.array(df.loc[df['sn_a_max'] == df['sn_a_j'],apds_err_a])*100/np.array(df.loc[df['sn_a_max'] == df['sn_a_j'],apds_j_a])
    df.loc[df['sn_b_max'] == df['sn_b_i'],apds_perr_b ]=np.array(df.loc[df['sn_b_max'] == df['sn_b_i'],apds_err_a])*100/np.array(df.loc[df['sn_b_max'] == df['sn_b_i'],apds_i_b])
    df.loc[df['sn_b_max'] == df['sn_b_j'],apds_perr_b ]=np.array(df.loc[df['sn_b_max'] == df['sn_b_j'],apds_err_a])*100/np.array(df.loc[df['sn_b_max'] == df['sn_b_j'],apds_j_b])
    df.loc[df['sn_b_max'] == df['sn_b_i'],apds_perr_r ]=np.array(df.loc[df['sn_b_max'] == df['sn_b_i'],apds_err_r])*100/np.array(df.loc[df['sn_b_max'] == df['sn_b_i'],apds_i_r])
    df.loc[df['sn_b_max'] == df['sn_b_j'],apds_perr_r ]=np.array(df.loc[df['sn_b_max'] == df['sn_b_j'],apds_err_r])*100/np.array(df.loc[df['sn_b_max'] == df['sn_b_j'],apds_j_r])

    # df.loc[df['sn_ar_max'] == df['sn_a_i_ratio'],apds_perr_a ]=np.array(df.loc[df['sn_ar_max'] == df['sn_a_i_ratio'],apds_err_a])*100/np.array(df.loc[df['sn_ar_max'] == df['sn_a_i_ratio'],apds_i_a])
    # df.loc[df['sn_ar_max'] == df['sn_a_j_ratio'],apds_perr_a ]=np.array(df.loc[df['sn_ar_max'] == df['sn_a_j_ratio'],apds_err_a])*100/np.array(df.loc[df['sn_ar_max'] == df['sn_a_j_ratio'],apds_j_a])
    # df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_perr_b ]=np.array(df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_err_a])*100/np.array(df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_i_b])
    # df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_perr_b ]=np.array(df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_err_a])*100/np.array(df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_j_b])
    # df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_perr_r ]=np.array(df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_err_r])*100/np.array(df.loc[df['sn_br_max'] == df['sn_b_i_ratio'],apds_i_r])
    # df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_perr_r ]=np.array(df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_err_r])*100/np.array(df.loc[df['sn_br_max'] == df['sn_b_j_ratio'],apds_j_r])


    df[apds_perr_r]=np.abs(df[apds_perr_r])
    df[apds_perr_b]=np.abs(df[apds_perr_b])
    df[apds_perr_a]=np.abs(df[apds_perr_a])
    df['APD_tot_a_dif'] = np.sum(df[apds_err_a], axis = 1)
    df['APD_tot_p_dif'] = np.sum(df[apds_perr_a], axis = 1)

    # df.loc[(df[apds_perr_a[2]]<10) & (df[apds_perr_a[4]]<5) & (df[apds_perr_a[9]]<5), 'APD_per_a_min'] = True
    # df['APD_per_a_min'].fillna(False, inplace=True)
    df['similarity'] = np.max(df[['APD_PError_b_3', 'APD_PError_b_5', 'APD_PError_b_7', 'APD_PError_b_9']].values, axis=1)


    df['d_ti'] = df['t_i'] - df['t_i'].shift(1)
    df['d_tj'] = df['t_j'] - df['t_j'].shift(1)
    df['d_t'] = abs(df['d_ti']-df['d_tj'] )/5000

    df['np_max'] =df.apply(lambda row: max(row['npower_a_j'], row['npower_a_i']), axis=1)
    df['sp_min'] =df.apply(lambda row: min(row['sp_a_j'], row['sp_a_i']), axis=1)
    df['amp_min'] =df.apply(lambda row: min(row['amp_a_j'], row['amp_a_j']), axis=1)

    df['sn_max'] =df.apply(lambda row: max(row['npower_a_j'], row['npower_a_i']), axis=1)
    df['sn_min'] =df.apply(lambda row: min(row['sp_a_j'], row['sp_a_i']), axis=1)

    df.loc[(df['APD_PError_a_5']<=10),'apd_ok']=1
    df.loc[(df['APD_PError_a_5']>10),'apd_ok']=0


    return df


def plot_cols2(df2, col, col3='sn_a_min',name_='raw', min_group_size=10):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=250)
    group_sizes = df2.groupby(df2.index).size()
    sorted_indices = group_sizes.sort_values(ascending=False).index
    sorted_df2 = pd.concat([df2.loc[idx] for idx in sorted_indices])
    grouped = sorted_df2.groupby(sorted_df2.index)
    sorted_grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
    filtered_grouped = [(name, group) for name, group in sorted_grouped if len(group) >= min_group_size]

    ax2 = ax.twinx()
    median_x = []
    median_y = []
    sem_y = []

    for i, (group_name, group_data) in enumerate(filtered_grouped):
        # Box plot
        bp = ax.boxplot(group_data[col], positions=[i+1], widths=0.8, vert=True, patch_artist=True, showfliers=False)

        for box in bp['boxes']:
            box.set_visible(False)
        for whisker in bp['whiskers']:
            whisker.set_visible(False)
        for cap in bp['caps']:
            cap.set_visible(False)
        for median in bp['medians']:
            median.set(color='red', linewidth=0.7)

        # Jittered scatter over box plot
        jitter = 0.7 * (np.random.rand(len(group_data[col])) - 0.5)
        ax.scatter(np.ones(len(group_data[col])) * (i+1) + jitter, group_data[col], alpha=0.5, color='black', s=0.3)

        # Get median of col3 and append to list
        median_col3 = group_data[col3].median()
        median_x.append(i+1)
        median_y.append(median_col3)

        sem_col3 = group_data[col3].std() / np.sqrt(len(group_data[col3]))
        sem_y.append(sem_col3)

    # Scatter plot for median col3 values with error bars
    ax2.errorbar(median_x, median_y, yerr=sem_y, fmt='o', color='blue', label=f'Median {col3}', markersize=1, marker='.', linestyle='dotted', linewidth=0.3, capsize=2)
    ax2.scatter(median_x, median_y, color='blue', s=1, marker='.')
    ax2.plot(median_x, median_y, color='blue', linestyle='dotted', linewidth=0.3)
    ax2.set_ylabel(col3, color='blue')
    ax2.tick_params(axis='y', colors='blue')

    # Set xticks and labels
    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([group[0].split('_')[0] for group in filtered_grouped], rotation=45, ha='right')
    ax.set_xticklabels([str(group[0]).split('_')[0] for group in filtered_grouped], rotation=45, ha='right')

    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([f'rec{i+1}' for i in range(len(filtered_grouped))], rotation=45, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    fig.savefig('results/'+name_+col+'_box'+".svg", format='svg')
    plt.show()
    
    
def plot_cols2(df2, col, col3='sn_a_min',name_='raw', min_group_size=10):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=250)
    group_sizes = df2.groupby(df2.index).size()
    sorted_indices = group_sizes.sort_values(ascending=False).index
    sorted_df2 = pd.concat([df2.loc[idx] for idx in sorted_indices])
    grouped = sorted_df2.groupby(sorted_df2.index)
    sorted_grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
    filtered_grouped = [(name, group) for name, group in sorted_grouped if len(group) >= min_group_size]

    ax2 = ax.twinx()
    median_x = []
    median_y = []
    sem_y = []

    for i, (group_name, group_data) in enumerate(filtered_grouped):
        # Box plot
        bp = ax.boxplot(group_data[col], positions=[i+1], widths=0.8, vert=True, patch_artist=True, showfliers=False)

        for box in bp['boxes']:
            box.set_visible(False)
        for whisker in bp['whiskers']:
            whisker.set_visible(False)
        for cap in bp['caps']:
            cap.set_visible(False)
        for median in bp['medians']:
            median.set(color='red', linewidth=0.7)

        # Jittered scatter over box plot
        jitter = 0.7 * (np.random.rand(len(group_data[col])) - 0.5)
        ax.scatter(np.ones(len(group_data[col])) * (i+1) + jitter, group_data[col], alpha=0.5, color='black', s=0.3)

        # Get median of col3 and append to list
        median_col3 = group_data[col3].median()
        median_x.append(i+1)
        median_y.append(median_col3)

        sem_col3 = group_data[col3].std() / np.sqrt(len(group_data[col3]))
        sem_y.append(sem_col3)

    # Scatter plot for median col3 values with error bars
    ax2.errorbar(median_x, median_y, yerr=sem_y, fmt='o', color='blue', label=f'Median {col3}', markersize=1, marker='.', linestyle='dotted', linewidth=0.3, capsize=2)
    ax2.scatter(median_x, median_y, color='blue', s=1, marker='.')
    ax2.plot(median_x, median_y, color='blue', linestyle='dotted', linewidth=0.3)
    ax2.set_ylabel(col3, color='blue')
    ax2.tick_params(axis='y', colors='blue')

    # Set xticks and labels
    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([group[0].split('_')[0] for group in filtered_grouped], rotation=45, ha='right')
    ax.set_xticklabels([str(group[0]).split('_')[0] for group in filtered_grouped], rotation=45, ha='right')

    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([f'rec{i+1}' for i in range(len(filtered_grouped))], rotation=45, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    fig.savefig('results/'+name_+col+'_box'+".svg", format='svg')
    plt.show()