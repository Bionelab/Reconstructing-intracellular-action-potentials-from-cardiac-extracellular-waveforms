import pickle
xg_cols  = ['t1','t2','ts','td' ]
iap_columns = ['APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60', 'APD70', 'APD80', 'APD90', 'APD100',]
eap_xg_columns =  [ 't1','t2','ts', 'td',  'IR', 'DR','v1_normalized', 'v2_normalized','vd_normalized', 'v1/v2','vd/v2', 'vd/v1', ]
all_eap_cols=['t1', 't2', 'ts', 't1+t2+td', 'td',  'v1_normalized', 'v2_normalized', 'v1/v2', 'vd/v1', 'vd/v2',  'vd_normalized', 'IR', 'DR', 's/n', ]
eap_eda_columns = ['t1', 't2','ts', 'td','IR', 'DR','v1', 'v2',  'vs','v1/v2', 'vd/v2',  'vd/v1',]
eap_xg_columns_screened = ['ts', 'td', 'IR_raw', 'DR_raw', 
                           # 'v1',
                           # 'v2',
                           # 'vd', 
                           'v1/v2', 'vd/v2', 'vd/v1']

# WE manualluy screenign and seperated the distorted spikes, for EDA and for XGBOOST --> needs to be automated soon
with open('data/distorted_dic_eda.pkl', 'rb') as f:
    distorted_dic_eda = pickle.load(f)
with open('data/distorted_dic_xgboost.pkl', 'rb') as f: # used for training
    distorted_dic_xg = pickle.load(f)
with open('data/distorted_dic_xg_dist.pkl', 'rb') as f: # used for distribution
    distorted_dic_xg_dist = pickle.load(f)

tests = ['exp00_dofetilide_W8_D2_i57_e47.pkl',
 'exp13_propranolol_W5_D1_i23_e13.pkl',
 'mea_2']
trains = ['exp00_dofetilide_W5_D1_i51_e52.pkl',
           'exp00_dofetilide_W8_D3_i24_e14.pkl',
           'exp00_dofetilide_W8_D3_i61_e62.pkl',
           'exp11_dofetilide_W5_D4_i23_e22_pace.pkl',
           'exp0_quinidine_W7_D6_i56_e68.pkl',
           'exp10_nifedipine_W5_D6_i82_e72_pace.pkl',
           'exp12_nifedipine_W5_D5_i43_e31_pace.pkl', #only 100
           'exp20_flecainide_W7_D2_i43_e42.pkl' ,# only 100
           'exp21_flecainide_W3_D10_i43_e53.pkl',
           'exp22_nifedipine_W5_D6_i55_e38_pace.pkl',# only 100
           'exp18_lidocaine_W8_D1_i14_e15.pkl',
           'exp29_nifedipine_W6_D10_i15_e34_pace.pkl',
           ]
# these recordings had good quality no need for second screening
to_exclude = ['exp13_propranolol_W5_D1_i23_e13.pkl','exp00_dofetilide_W8_D3_i24_e14.pkl',]
all_keys = trains+tests
#XGboost
best_params = {'lambda': 5.1070957448797305,
 'alpha': 0.0320563434904784,
 'min_child_weight': 12,
 'max_depth': 9,
 'subsample': 0.6}
