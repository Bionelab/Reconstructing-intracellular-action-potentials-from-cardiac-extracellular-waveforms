import glob
UNSEEN = ['W8FebD2_10nM_Dofe_57_47_8000']
ALL_KEYS = ['W5FebD1_10nM_Dofe_51_52_8000', 'W5FebD1_10nM_Dofe_57_58_8000', 'W5FebD1_10nM_Dofe_84_85_8000', 'W8FebD3_10nM_Dofe_24_14_8000', 'W8FebD3_10nM_Dofe_61_62_8000', 'W8FebD2_10nM_Dofe_57_47_8000']
TRAIN_KEYS = ['W5FebD1_10nM_Dofe_51_52_8000', 'W5FebD1_10nM_Dofe_57_58_8000', 'W5FebD1_10nM_Dofe_84_85_8000', 'W8FebD3_10nM_Dofe_24_14_8000', 'W8FebD3_10nM_Dofe_61_62_8000',]

#Smoothing the final predictions
smoothing = False 
lens1 = 201
lens2 = 1001
pol1 = 3
pol2 = 3
lenghts = 8000
########

## UNET MODEL HYPERPARAMETERS
UNET_HYPERPARAMS={'loss_data':'mae',
'seed' : 5,
'kernel_size':13,
'ch_num':32,
'depth':8,
'b_s':32,
'w_mae':10,
'wph':0.04,	
'lr':0.0025,	
"epoch":100,
'physics' : True,
                 }


#######
fs = 5000
lens1 = 201
# lens2 = 1001
pol1 = 3
pol2 = 3
times = [(0, 410000),(420000, 1200000), (1300000, 5515000)]
lens2 = 1501
lowcut   = 0.125
highcut  = fs/32
lowcut2  = 0.1
highcut2 = fs/2-1
ds = 4000
ps = 30
order = 3
#####
# for mae
selected_model= 'models/ph_True_seed3.h5'
selected_q_model= 'model_quantile/s_510.05_lr_0.0025.h5'
rmse_models = glob.glob('models/*.h5')
