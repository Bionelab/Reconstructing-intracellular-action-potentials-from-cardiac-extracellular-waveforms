from constants import *
from utils import*
from load_recording_data import *
seed=42

data_feature = pd.read_csv('models/data_feat.csv')

with open('eap_iap_data.pkl', 'rb') as inp:
  all_recordings = pickle.load(inp)

data={}

for key in ALL_KEYS:
    print(key) 
    d_8 =  MYDataset(all_recordings, [key], function1_min_0, function2, test_size=0.0001, val_size = 0.001, seedd = seed  )
    data[(key,'extra')]=d_8.extras_normalized
    data[(key,'intra')]=d_8.intras_normalized
    data[(key,'extra_raw')]=d_8.extras_raw
    data[(key,'intra_raw')]=d_8.intras_raw


intras_test,extras_test =data_prep_test (data,UNSEEN,raw = False)
intras_train,extras_train =data_prep_test (data,TRAIN_KEYS,raw = False)
intras_test_raw,extras_test_raw =data_prep_test (data,UNSEEN,raw = True)
intras_train_raw,extras_train_raw =data_prep_test (data,TRAIN_KEYS,raw = True)
apds_test =  get_all_apds(intras_test)
apds_train=  get_all_apds(intras_train)
