import optuna
import optuna.integration.lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor

def objective(trial, all_train, df_train):
    param = {
        # 'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        # 'sampling_method': 'gradient_based',
        'lambda': trial.suggest_float('lambda', 0.01, 50.0, log=True),
        'alpha': trial.suggest_float('alpha', 0.01, 50.0, log=True),
        'eta': trial.suggest_float('alpha', 0.01, 50.0, log=True),
        'gamma':trial.suggest_float('alpha', 0.01, 50.0, log=True),
        # 'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        # 'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 4, 50),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 200),
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7,8,9,10]),
        'subsample': trial.suggest_categorical('subsample', [0.2,0.5,0.6,0.7,0.8,1.0]),
        'random_state': 51,
        # 'early_stopping_rounds': 10
    }

    total_error = 0
    for name in all_train:
        # df_train = df_traine[df_traine['name'].isin(name)]  # Select rows where 'name' is in the current list
        # df_val = df_traine[~df_traine['name'].isin(name)]  
        df_train = df_traine[df_traine['name']!=name]  # Select rows where 'name' is in the current list
        df_teste = df_traine[df_traine['name']==name] 
        X_train, X_val, y_train, y_val = train_test_split(df_train[col2], df_train[col_pred]/5000, test_size=0.3, random_state=42)
        model = XGBRegressor(**param,early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)
        preds = model.predict(df_teste[col2])
        error = mean_squared_error(df_teste[col_pred]/5000, preds)
        total_error += error
    # print(total_error)
    return total_error  / len(df_traine['name'].unique())


import matplotlib.pyplot as plt
import shap