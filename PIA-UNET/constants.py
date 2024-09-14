import pickle
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
# to avoid overfitting we limited the number of samples used for these recordings to max 350 samples each
max_samples = 350
trains_with_limits = ['exp22_nifedipine_W5_D6_i55_e38_pace.pkl',
                      'exp12_nifedipine_W5_D5_i43_e31_pace.pkl',
                      'exp20_flecainide_W7_D2_i43_e42.pkl']

UNET_HYPERPARAMS={'loss_data':'mae',
'seed' : 91,
'kernel_size':11,
'ch_num':32,
'depth':8,
'b_s':32,
'w_mae':10, # data
'wph':0.05,	# physics
'lr':0.001,	
"epoch":120,
'physics' : True,
'max_sample': 350, # to not get overly trained on some recordings with a lot of samples
'use_com': False,  # using bottle neck or the last layer for physics
'use_cond': False,  } # using condition on applying physics on u > 0.1

