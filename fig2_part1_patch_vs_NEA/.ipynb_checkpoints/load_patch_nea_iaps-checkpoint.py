import os
import numpy as np
import re

def extract_number(filename):
    # Extract the first number from the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def load_data(base_path):
    # Initialize lists for n arrays, p arrays, and names
    n_arrays = []
    p_arrays = []
    names = []

    # List all .npy files in the directory and sort them
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.npy')])

    # Separate the filenames into n and p groups with corresponding indices
    n_files = [f for f in files if f.startswith('n')]
    p_files = [f for f in files if f.startswith('p')]

    # Ensure nx is at the same index as px
    n_files, p_files = zip(*sorted(zip(n_files, p_files), key=lambda x: extract_number(x[0])))

    # Load each file and append to the appropriate list
    for n_file, p_file in zip(n_files, p_files):
        n_data = np.load(os.path.join(base_path, n_file))
        p_data = np.load(os.path.join(base_path, p_file))
        n_arrays.append(n_data)
        p_arrays.append(p_data)
        names.append(n_file[:-4])  # Append the name without '.npy'

    return n_arrays, p_arrays, names


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
        
        p__ = p4_/np.max(p4_)
        n__ = n4_/np.max(n4_)
        plt.figure(figsize = (20,2))
        plt.plot(p__, color ='green')
        plt.plot(p_peaks2,p__[p_peaks2], 'ro')
        plt.plot(n__, color ='blue')
        plt.plot(n_peaks2,n__[n_peaks2], 'bo')
        plt.xlim(0,700000)
        plt.show()
        
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
                            window_patch = p4_[tp-ds-2000:tp-ds+6000]
                            window_nea   = n4_[tn-2000:tn+6000]
                            window_noise_patch = noise_p4[tp-ds-2000:tp-ds+6000]
                            window_noise_nea   = noise_n4[tn-2000:tn+6000]     
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
                            if '_sh' in name:
                                apds_patch_, l_patch = get_apd(window_patch[:6000])
                                apds_nea_, l_nea= get_apd(window_nea[:6000])
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
                            if counter_each == 10:
                                plt.plot(window_patch_norm)
                                plt.plot(window_nea_norm)
                                plt.show()
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
    return dic_patch_nea, df
                                                
