import pickle
seed = 42

keys = [
       'W5FebD1_10nM_Dofe_51_52_8000',
    'W5FebD1_10nM_Dofe_57_58_8000', 
    'W5FebD1_10nM_Dofe_84_85_8000',
    'W8FebD3_10nM_Dofe_24_14_8000',
    'W8FebD3_10nM_Dofe_61_62_8000',
    'W8FebD2_10nM_Dofe_57_47_8000']

train_keys= ['W5FebD1_10nM_Dofe_51_52_8000',
 'W5FebD1_10nM_Dofe_57_58_8000',
 'W5FebD1_10nM_Dofe_84_85_8000',
 'W8FebD3_10nM_Dofe_24_14_8000',
 'W8FebD3_10nM_Dofe_61_62_8000']

train_keys2 =[['W5FebD1_10nM_Dofe_51_52_8000',
             'W5FebD1_10nM_Dofe_57_58_8000',
             'W5FebD1_10nM_Dofe_84_85_8000'],
             ['W8FebD3_10nM_Dofe_24_14_8000',
             'W8FebD3_10nM_Dofe_61_62_8000']]

UNSEEN8_a = ['W8FebD2_10nM_Dofe_57_47_8000'] # test set
col_pred =  ['APD'+str(10*i) for i in range(1,11) ]

distorted={
    'W5FebD1_10nM_Dofe_51_52_8000':[i for i in range(24,38) ]+[i for i in range(42,186) ]+[i for i in range(276,284)],
    'W5FebD1_10nM_Dofe_57_58_8000':[0,13,15,18,19,20,21,22,23,24,25,26,27]+[i for i in range(126,243) ],
    'W5FebD1_10nM_Dofe_84_85_8000':[i for i in range(0,32) ],
    'W8FebD3_10nM_Dofe_24_14_8000':[6]+[i for i in range(14,89)],
    'W8FebD3_10nM_Dofe_61_62_8000':[i for i in range(0,45)]+[i for i in range(55,369)],
    'W8FebD2_10nM_Dofe_57_47_8000':[]}


eap_eda_columns = ['w1', 'w2','w1+w2', 'w3','r3', 'd3','h1', 'h2',  'h3','h1/h2', 'h3/h2',  'h3/h1',]
eap_xg_columns = [ 'w1+w2',  'w3',  'r3', 'd3','h1', 'h2', 'h1/h2','h3','h3/h2', 'h3/h1' ]
eap_xg_columns_screened = [ 'w1+w2',  'w3',  'r3', 'd3','h1', 'h2', 'h1/h2','h3/h2', 'h3/h1' ]
iap_columns = ['APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60', 'APD70', 'APD80', 'APD90', 'APD100',]

eap_timeamp_dep_slope_col = ['r3', 'd3']
eap_time_dep_cols = ['w1+w2',  'w3']
eap_amp_cols = ['h1', 'h2',  'h3',]

xg_cols  = ['w1+w2', 'w3+w1+w2', 'w3', 'h1',
       'h2', 'h1/h2', 'h3/h1', 'h3/h2', 'h3','r3', 'd3',
       'APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60', 'APD70', 'APD80',
       'APD90', 'APD100', ]


best_params = {'lambda': 14.992478628864632,
 'alpha': 0.13558743319133845,
 'n_estimators': 48,
 'min_child_weight': 227,
 'max_depth': 9,
 'subsample': 0.7}


# best_params= {'lambda': 0.19932669695805677,
#  'alpha': 0.0923274647535503,
#  'min_child_weight': 172,
#  'max_depth': 8,
#  'subsample': 0.5}