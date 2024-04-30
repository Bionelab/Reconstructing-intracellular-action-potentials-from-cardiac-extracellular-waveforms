import os
import pickle
import pandas as pd
#####
DATA_FOLDER = 'data_iap'
F1_AVA_FILE_NAME ='f1_availabiliteis.csv'
F2_AVA_FILE_NAME ='f2_availabiliteis.csv'
RAW_DATA_NAME = 'data_raw_neighboring_iap.pkl'
RAW_SINGLE_CHANNEL_DATA_NAME = 'dic_data_single_channel.pkl'
###########
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, DATA_FOLDER)
f1_av= pd.read_csv( os.path.join(data_directory, F1_AVA_FILE_NAME), index_col = 0)
f2_av= pd.read_csv( os.path.join(data_directory, F2_AVA_FILE_NAME), index_col = 0)
with open( os.path.join(data_directory, RAW_DATA_NAME), 'rb') as file:
    data_raw = pickle.load(file)
with open( os.path.join(data_directory, RAW_SINGLE_CHANNEL_DATA_NAME), 'rb') as file:
    dic_data_single_channel = pickle.load(file)

###
t_f2 = [[0,260000],[400000, 2100000],[2500000, 4200000],[4500000, 6400000],[6750000, 8500000],[8800000, 11178998]]
t_f1 = [[100000,340000],[790000, 1500000],[1700000, 2000000],[2200000, 4000000],[4300000, 6100000],[6400000, 8199999]]
####
fs = 5000.0   
order = 3
lowcut_iap = 0.125
highcut_iap  = fs/32
highcut = highcut_iap
lowcut_iap2 = 0.1
highcut_iap2 = fs/2-1
#
ps_iap = 1000
ds = 7000
###
#column names
apds_i_a = ['APD_i_a_'+str(i) for i in range(1,11)]
apds_j_a= ['APD_j_a_'+str(i) for i in range(1,11)]
apds_err_a = ['APD_Error_a_'+str(i) for i in range(1,11)]
apds_perr_a = ['APD_PError_a_'+str(i) for i in range(1,11)]

#second filter
apds_i_b = ['APD_i_b_'+str(i) for i in range(1,11)]
apds_j_b= ['APD_j_b_'+str(i) for i in range(1,11)]
apds_err_b = ['APD_Error_b_'+str(i) for i in range(1,11)]
apds_perr_b = ['APD_PError_b_'+str(i) for i in range(1,11)]
#raw data
apds_i_r = ['APD_i_r_'+str(i) for i in range(1,11)]
apds_j_r= ['APD_j_r_'+str(i) for i in range(1,11)]
apds_err_r = ['APD_Error_r_'+str(i) for i in range(1,11)]
apds_perr_r = ['APD_PError_r_'+str(i) for i in range(1,11)]

apd_i_col = ['APD_i_ '+ str(i) for i in range(1,11)]
apd_j_col = ['APD_j_'+ str(i) for i in range(1,11)]
apd_dif_col  = ['APD Error'+ str(i) for i in range(1,11)]
apd_perc_err = ['APD_p_'+ str(i) for i in range(1,11)]

#####
baseline_channel = {'f1':('f1',20, 21),
                 'f2':('f2',9, 10)}
times = {'f1':t_f1,
          'f2':t_f2}

availablities= {'f1':f1_av,
                'f2':f2_av}
#########
eap_cols=[ 'w1', 'w2', 'w1+w2', 'w3', 'r3', 'd3',
       'h1', 'h2','h3',  'h1/h2',  'h3/h2', 'h3/h1',]
iap_cols=[ 'APD10', 'APD20','APD30', 'APD40', 'APD50', 'APD60', 'APD70', 'APD80', 'APD90', 'APD100']

xg_cols  = ['w1+w2', 'w3+w1+w2', 'w3', 'h1',
       'h2', 'h1/h2', 'h3/h1', 'h3/h2', 'h3','r3', 'd3',
       'APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60', 'APD70', 'APD80',
       'APD90', 'APD100', ]