# -*- coding: utf-8 -*-
# pip install McsPyDataTools
# pip install tsaug
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import h5py as h5
import glob
import tsaug
import tensorflow as tf
from keras.layers import Dense, Dropout, Reshape, Conv1D,Conv2D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply, Flatten, LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from keras.models import load_model
import random
from utils import *
from constants import *


def cbr(x, out_layer, kernel, stride, dilation, activation ="relu", pad="same"  ):
    x = tf.keras.layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding=pad,
               kernel_initializer=tf.keras.initializers.HeNormal())(x)
    # x = tf.keras.layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding=pad,
    #            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def se_block(x_in, layer_n, x_out):
    x = tf.keras.layers.GlobalAveragePooling1D()(x_in)
    x = tf.keras.layers.Dense(layer_n//8, activation="relu",)(x )
    x = tf.keras.layers.Dense(layer_n, activation="sigmoid")(x )
    x_out=tf.keras.layers.Multiply()([x, x_out])
    # x_out = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([x, x_out])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se):
    x1 = cbr(x_in, layer_n, kernel, 1, dilation)
    x2 = cbr(x1, layer_n, kernel, 1, dilation,activation ="sigmoid")
    if use_se:
        x2 = se_block(x_in, layer_n, x2)
    x = tf.keras.layers.Add()([x_in, x2])
    return x  

def att(g,shortcut): #attention block edited
    
    n_layer = shortcut.shape[2]
    x1 =    tf.keras.layers.Conv1D(n_layer,kernel_size = 1,strides = 5, 
                  dilation_rate=1,  padding='same',kernel_initializer=tf.keras.initializers.HeNormal())(shortcut)
    # tf.keras.initializers.GlorotNormal(seed=None)
    # 

    print(x1.shape, 'x1 on shortcut')
    g1 =    tf.keras.layers.Conv1D(n_layer,kernel_size = 1,strides = 1, padding='same',)(g) 
    
    print(g1.shape, 'g1 on g')
    
    g1_x1 =  tf.keras.layers.Add()([g1,x1])
    print(g1_x1.shape, 'g1_x1')
    psi =  tf.keras.layers.Activation('relu')(g1_x1)
    psi = tf.keras.layers.Conv1D(1,kernel_size = 1,padding='same',   )(psi) 
    print(psi.shape, 'psi: conv on psi')
    psi =  tf.keras.layers.Activation('sigmoid')(psi)
    psi =  tf.keras.layers.UpSampling1D(5)(psi)
    print(psi.shape, 'psi_2 : psi_upsample5')
    x =  tf.keras.layers.Multiply()([shortcut,psi])
    print(x.shape, 'x_psi * shortcut')
    # print('att x*psi',psi.shape, shortcut.shape, x.shape)
    return x


def physics_part(input_layer,v_out,b,name):
       
        a_out = tf.keras.layers.Dense(1)(b )
        a_out= tf.keras.layers.ReLU(name = 'a'+name)(a_out)
        a_out= tf.keras.activations.sigmoid(a_out)
        a_oot = a_out
        a_out = tf.expand_dims(a_out, 1)
        a_out = tf.broadcast_to(a_out , [tf.keras.backend.shape(input_layer )[0], tf.keras.backend.shape(input_layer )[1],1] )
        print('a3 ', a_out .shape)

        k_out= tf.keras.layers.Dense(1)(b )
        k_out= tf.keras.layers.ReLU(name = 'k'+name)(k_out)
        k_oot = k_out
        k_o2 = tf.expand_dims(k_out, 1)
        print('k2', k_o2.shape)
        k_o3 = tf.broadcast_to(k_o2 , [tf.keras.backend.shape(input_layer )[0],tf.keras.backend.shape(input_layer )[1],1] )
        print('k_o3 ', k_o3 .shape)

        ones = tf.ones_like(v_out)
        epsilon = 1e-3
        # v_out_noisy = tf.where(tf.equal(v_out, 0), tf.ones_like(v_out) * epsilon, v_out)
        v_out_noisy = tf.where(tf.equal(v_out, 0), tf.ones_like(v_out) * epsilon, v_out)
    
        v_out2 = v_out_noisy
        print('v_out2', v_out2.shape)
    
        dv_l = tf.keras.layers.ZeroPadding1D(padding=(2,0), name ='dv_l'+name)(v_out_noisy)
        dv_r = tf.keras.layers.ZeroPadding1D(padding=(0,2), name ='dv_r'+name)(v_out_noisy)
        dv_t = (dv_r - dv_l)[:,2:-2,:]
        dv_t = tf.keras.layers.ZeroPadding1D(padding=(1,1), name ='dv_t'+name)( dv_t)
        dv_t = (dv_t*2*(1/12.9))
        
        dv2_t = (dv_r + dv_l)[:,2:-2,:] 
        dv2_t = tf.keras.layers.ZeroPadding1D(padding=(1,1), name ='dv2_t'+name)( dv2_t)
        dv2_t = (dv2_t - 2*v_out_noisy) *(1/12.9)**2
        
        print('dv_t'+name, dv_t.shape)
        print('dv2_t'+name, dv2_t.shape)
        print('ones'+name, ones.shape)
        
        a_v =  tf.keras.layers.Multiply(name = 'a_v'+name)([a_out,v_out_noisy])
        inv_v = v_out_noisy ** -1
        inv_v2 = v_out_noisy ** -2
        
        inv_v = tf.where(tf.math.is_nan(inv_v), tf.fill(tf.shape(inv_v), 0.0), inv_v)
        inv_v2 = tf.where(tf.math.is_nan(inv_v2), tf.fill(tf.shape(inv_v2), 0.0), inv_v2)
        
        inv_v = tf.where(tf.logical_or(tf.math.is_nan(inv_v), tf.math.is_nan(inv_v2)), tf.zeros_like(inv_v, dtype=inv_v.dtype),  inv_v)
        inv_v2 = tf.where(tf.logical_or(tf.math.is_nan(inv_v), tf.math.is_nan(inv_v2)), tf.zeros_like(inv_v2, dtype=inv_v2.dtype),  inv_v2)

        
        print('inv_v', inv_v.shape, inv_v )
        print('inv_v2', inv_v2.shape, inv_v2)
        print('a_v', a_v.shape, a_v)

        term_l1 = tf.keras.layers.Multiply(name ='tl1'+name)([k_o3 ,  dv_t, ones - 2*v_out_noisy+a_out])
        term_l1 = tf.where(tf.logical_or(tf.math.is_nan(inv_v), tf.math.is_nan(inv_v2)), tf.zeros_like(term_l1, dtype=term_l1.dtype),  term_l1)
        
        print('term_l1', term_l1)
        term_l2 = tf.keras.layers.Multiply(name='tl2'+name)([inv_v2 , dv_t**2])
        print('term_l2', term_l2)
        term_l3 = tf.keras.layers.Multiply(name='tlt'+name)([inv_v, dv2_t])
        print('term_l3', term_l3)
        term_lt = term_l1 + term_l2 - term_l3
        

        print('term_l1', term_l1.shape)
        print('term_l2', term_l2.shape)
        print('term_l3', term_l3.shape)
        print(term_lt.shape)
        
        step = 0.9 *(tf.keras.activations.sigmoid(10*(-v_out_noisy+a_out))) + (ones * 0.1)
        step = tf.where(tf.logical_or(tf.math.is_nan(inv_v), tf.math.is_nan(inv_v2)), tf.zeros_like(step, dtype=step.dtype),  step)
        
        term_r1 = tf.keras.layers.Multiply(name='tr1'+name)([k_o3, v_out_noisy ** 2 + a_out - a_v])
        term_r1 = tf.where(tf.logical_or(tf.math.is_nan(inv_v), tf.math.is_nan(inv_v2)), tf.zeros_like(term_r1, dtype=step.dtype),  term_r1)
        
        term_r2 = tf.keras.layers.Multiply(name='tr2'+name)([dv_t, inv_v ])
        term_rt = tf.keras.layers.Multiply(name='trt'+name)([step, term_r1 + term_r2 ])

        dv_out = term_lt -term_rt
        epssss = 1e-5
        dv_out = tf.math.log(dv_out**2+epssss)
        # dv_out_max =  tf.reduce_max(tf.abs(dv_out), axis=1)
        dv_out_max = dv_out
        # dv_out_max = dv_out
        print('dv_out',dv_out.shape)
        # print('dv_out_max',dv_out_max.shape)

        return dv_out_max


def Unet(input_shape=(8000,1), layer_n =64,kernel_size = 7, depth = 4, physics = True,use_se = True, use_com = True ): #att_n
    tf.keras.backend.clear_session()
    layer_n = layer_n
    kernel_size = kernel_size 
    depth = depth
    input_layer = tf.keras.Input(input_shape)   
    tf.keras.backend.clear_session()
    # print(input_layer.shape)
    input_layer_1 = tf.keras.layers.AveragePooling1D(5,padding='same')(input_layer)
    input_layer_1 =cbr(input_layer_1, 16, kernel_size, 1, 1)
    input_layer_1 =cbr(input_layer_1 , 16, kernel_size, 1, 1,activation ="sigmoid")
    # input_layer_2 = tf.keras.layers.AveragePooling1D(25,padding='same')(input_layer)
    # input_layer_2 =cbr(input_layer_2 , 16, kernel_size, 1, 1,activation ="sigmoid")
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000
    s1 = x.shape
    print('cbr1:',s1, layer_n)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1,use_se)
    out_0 = x
    s11 = out_0.shape
    print('re1:',s11)
    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    s2 = x.shape
    print('cbr2:',s2,layer_n*2)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1,use_se)
    out_1 = x
    s3 = x.shape
    print('re2:',s3)
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    s5 = x.shape
    print('cbr3:',s5,layer_n*3)  
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1,use_se)
    out_2 = x
    s6 = x.shape
    print('re3:',s6)  
    print('last_layer')
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    s8 = x.shape
    print('cbr4:',s8,layer_n*4)  
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1,use_se)
    x_comp = x
    s9 = x.shape
    print('re4:',s9)  
    ########### Decoder
    us = 5
    out_2_at = att(x,out_2)
    print('pre_co_-3',  tf.keras.layers.UpSampling1D(us)(x),x.shape ,out_2_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_2_at])
    d2 = x.shape
    print('aft_con-3',d2) 
    x = cbr(x, layer_n*3, kernel_size, 1, 1)
    d3 = x.shape
    print('cbr-4:',d3) 
    d4 = x.shape
    print('upS2 with',us,":",d4)
    out_1_at = att(x,out_1)
    print('out_1_at', out_1_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_1_at])
    d5 = x.shape
    print('d5', d5)
    x = cbr(x, layer_n*2, kernel_size, 1, 1)
    d6 = x.shape
    print('cbr-3:',d6) 
    d7 = x.shape
    print('upS3 with',us,":",d7)
    out_0_at = att(x,out_0)
    print('pre_co_-2', x.shape ,out_0_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_0_at])
    x = cbr(x, layer_n, kernel_size, 1, 1)    
    d8 = x.shape
    print('aft_con-1',d8)
    v = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
    v_out = tf.keras.layers.Activation("tanh", name = 'v')(v)
    print(v_out.shape)
    if physics == True:
        print('ph')
        if use_com==True:
            print('Comp_ON')   
            # bk = cbr(v_out, 1, kernel_size, 1, 1)
            bk = cbr(x_comp, 1, kernel_size, 1, 1)
        else:
            bk = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
        b = tf.keras.layers.Flatten()(bk)
        print('b',b.shape)
        b= tf.keras.layers.Dense(8,kernel_initializer= tf.keras.initializers.HeNormal())(b )
        b= tf.keras.layers.Dropout(0.4)(b)
        print('b_1',b.shape)  
        b = tf.keras.layers.BatchNormalization()(b)
        b=  tf.keras.layers.LeakyReLU(alpha=0.1)(b)
        print('b_1',b.shape)
        b= tf.keras.layers.Dropout(0.4)(b)
        dv_out_max = physics_part(input_layer,v_out,b,'noq')
        model =  tf.keras.Model(input_layer, [v_out, dv_out_max])
    else:
        model =  tf.keras.Model(input_layer, v_out)
    return model 




def b_produce(x_comp,kernel_size):
        bk = cbr(x_comp, 1, kernel_size, 1, 1)
        b = tf.keras.layers.Flatten()(bk)
        b= tf.keras.layers.Dense(8,kernel_initializer= tf.keras.initializers.HeNormal())(b )
        b= tf.keras.layers.Dropout(0.4)(b)
        b = tf.keras.layers.BatchNormalization()(b)
        b=  tf.keras.layers.LeakyReLU(alpha=0.1)(b)
        b= tf.keras.layers.Dropout(0.4)(b)
        return b

def Unet_qt_all(input_shape=(8000,1), layer_n =64,kernel_size = 7, depth = 4, physics = True,use_se = True, use_com = True ): #att_n
    tf.keras.backend.clear_session()
    layer_n = layer_n
    kernel_size = kernel_size 
    depth = depth
    input_layer = tf.keras.Input(input_shape)   
    tf.keras.backend.clear_session()
    # print(input_layer.shape)
    input_layer_1 = tf.keras.layers.AveragePooling1D(5,padding='same')(input_layer)
    input_layer_1 =cbr(input_layer_1, 16, kernel_size, 1, 1)
    input_layer_1 =cbr(input_layer_1 , 16, kernel_size, 1, 1,activation ="sigmoid")
    # input_layer_2 = tf.keras.layers.AveragePooling1D(25,padding='same')(input_layer)
    # input_layer_2 =cbr(input_layer_2 , 16, kernel_size, 1, 1,activation ="sigmoid")
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000
    s1 = x.shape
    print('cbr1:',s1, layer_n)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1,use_se)
    out_0 = x
    s11 = out_0.shape
    print('re1:',s11)
    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    s2 = x.shape
    print('cbr2:',s2,layer_n*2)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1,use_se)
    out_1 = x
    s3 = x.shape
    print('re2:',s3)
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    s5 = x.shape
    print('cbr3:',s5,layer_n*3)  
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1,use_se)
    out_2 = x
    s6 = x.shape
    print('re3:',s6)  
    print('last_layer')
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    s8 = x.shape
    print('cbr4:',s8,layer_n*4)  
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1,use_se)
    x_comp = x
    s9 = x.shape
    print('re4:',s9)  
    ########### Decoder
    us = 5
    out_2_at = att(x,out_2)
    print('pre_co_-3',  tf.keras.layers.UpSampling1D(us)(x),x.shape ,out_2_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_2_at])
    d2 = x.shape
    print('aft_con-3',d2) 
    x = cbr(x, layer_n*3, kernel_size, 1, 1)
    d3 = x.shape
    print('cbr-4:',d3) 
    d4 = x.shape
    print('upS2 with',us,":",d4)
    out_1_at = att(x,out_1)
    print('out_1_at', out_1_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_1_at])
    d5 = x.shape
    print('d5', d5)
    x = cbr(x, layer_n*2, kernel_size, 1, 1)
    d6 = x.shape
    print('cbr-3:',d6) 
    d7 = x.shape
    print('upS3 with',us,":",d7)
    out_0_at = att(x,out_0)
    print('pre_co_-2', x.shape ,out_0_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_0_at])
    x = cbr(x, layer_n, kernel_size, 1, 1)    
    d8 = x.shape
    print('aft_con-1',d8)
    v1 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
    v_out_q1 = tf.keras.layers.Activation("tanh", name = 'vq1')(v1)
    v2 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
    v_out_q2 = tf.keras.layers.Activation("tanh", name = 'vq2')(v2)
    v3 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
    v_out_q3 = tf.keras.layers.Activation("tanh", name = 'vq3')(v3)


    b1 =  b_produce(x_comp,kernel_size)
    b2 =  b_produce(x_comp,kernel_size)
    b3 =  b_produce(x_comp,kernel_size)
    dv_out_max_q1 = physics_part(input_layer,v_out_q1,b1,'q1')
    dv_out_max_q2 = physics_part(input_layer,v_out_q2,b2,'q2')
    dv_out_max_q3 = physics_part(input_layer,v_out_q3,b3,'q3')
    model =  tf.keras.Model(input_layer, [v_out_q1,v_out_q2,v_out_q3, dv_out_max_q1,dv_out_max_q2,dv_out_max_q3])
    return model 


def scheduler(epoch, lr):
  if epoch < 25:
        return lr
  else:
        return lr * tf.math.exp(-0.1)


## no physics
def model_fit(model, 
              train_input1,
              val_input1,
              train_target1,
              val_target1,
               n_epoch, batch_size=32):
        
        callbacks_list = [      tf.keras.callbacks.LearningRateScheduler(scheduler),   
                            # tf.keras.callbacks.EarlyStopping(
                            #                     monitor="val_loss",
                            #                     patience=40,
                            #                     mode="auto",
                            #             restore_best_weights=True, )
                         ]
        
        history = model.fit(x=train_input1, y=train_target1,
                              validation_data=(val_input1, val_target1),
                              batch_size= batch_size,
                              epochs=n_epoch,
                              shuffle=True, verbose=2,
                              callbacks=callbacks_list)
        
        # model.load_weights(fn)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.show()
        return history, model

# Define your learning rate scheduler function

class NaNHandlingCallback(tf.keras.callbacks.Callback):
    def __init__(self, scheduler_function, factor=0.5, min_lr=1e-6, history_size=5):
        super(NaNHandlingCallback, self).__init__()
        self.weights_history = []  # List to store weights history
        self.loss_history = []     # List to store loss history
        self.scheduler_function = scheduler_function
        self.factor = factor
        self.min_lr = min_lr
        self.history_size = history_size

    def on_train_begin(self, logs=None):
        self.weights_history.clear()
        self.loss_history.clear()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        current_weights = self.model.get_weights()

        # Store current weights and loss at the end of each epoch
        self.weights_history.append(current_weights)
        self.loss_history.append(current_loss)

        # Keep only the last 'history_size' sets of weights and losses
        if len(self.weights_history) > self.history_size:
            self.weights_history.pop(0)
            self.loss_history.pop(0)

        # Check for NaN or Inf in loss or weights
        if np.isnan(current_loss) or np.isinf(current_loss) or any(np.isnan(w).any() for w in current_weights):
            print(f"NaN or Inf detected in loss or weights at epoch {epoch}.")
            # Restore weights from the best epoch based on saved loss history
            best_epoch = np.argmin(self.loss_history[:-1])  # Exclude the current epoch
            self.model.set_weights(self.weights_history[best_epoch])
            print(f"Restored model weights from epoch {best_epoch}.")
            
            # Reduce learning rate
            new_lr = max(self.model.optimizer.lr * self.factor, self.min_lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"Reduced learning rate to {new_lr}")            
            # self.model.stop_training = True

    def on_epoch_begin(self, epoch, logs=None):
        # Adjust learning rate using the scheduler function
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        scheduled_lr = self.scheduler_function(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


#with physics
def model_fit_physics(model, 
              train_input1,
              val_input1,
              train_target1,
              train_target2 ,
              val_target1,
              val_target2,
               n_epoch, batch_size=32):


    callbacks_list = [      NaNHandlingCallback(scheduler),]
    history = model.fit(x=train_input1, y=[train_target1, train_target2,],
                              validation_data=(val_input1, [val_target1,val_target2]),
                              batch_size= batch_size,
                              epochs=n_epoch,
                              shuffle=True, verbose=2,
                              callbacks=callbacks_list)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()
    return history, model

#somple
def model_fit_physics_qt(model, 
              train_input1,
              val_input1,
              train_target1,
              train_target2 ,
              val_target1,
              val_target2,
               n_epoch, batch_size=32):


    callbacks_list = [      NaNHandlingCallback(scheduler),]
    history = model.fit(x=train_input1, y=[train_target1, train_target2],
                              validation_data=(val_input1, [val_target1,val_target2]),
                              batch_size= batch_size,
                              epochs=n_epoch,
                              shuffle=True, verbose=2,
                              callbacks=callbacks_list)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()
    return history, model

def model_fit_physics_qt_all(model, 
              train_input1,
              val_input1,
              train_target1,
              train_target2 ,
              val_target1,
              val_target2,
               n_epoch, batch_size=32):


    callbacks_list = [      NaNHandlingCallback(scheduler),]
    history = model.fit(x=train_input1, y=[train_target1,train_target1,train_target1, train_target2,train_target2,train_target2],
                              validation_data=(val_input1, [val_target1,val_target1,val_target1,val_target2,val_target2,val_target2]),
                              batch_size= batch_size,
                              epochs=n_epoch,
                              shuffle=True, verbose=2,
                              callbacks=callbacks_list)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()
    return history, model




def train_model (data,train_sets, test_sets, **kwargs):
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    tf.keras.backend.clear_session()
    
    intras_test,extras_test =data_prep_test (data,test_sets)
    intras_train,extras_train =data_prep_test (data,train_sets)
    model = Unet(input_shape=(lenghts,1), layer_n =kwargs['ch_num'] ,kernel_size = kwargs['kernel_size'], depth = kwargs['depth'], physics=kwargs['physics'],use_com=True )
    if kwargs['physics'] == False:
        model.compile(
                            loss=[ kwargs['loss_data']],
                                    optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                    metrics=[
                                            "mean_absolute_error",  
                                            ])
        hist, model = model_fit(model,
                                      extras_train,
                                      extras_test,
                                      intras_train,
                                      intras_test,
                                  kwargs['epoch'], 
                                 kwargs['b_s'] )       

        plt.plot(hist.history['loss'][50:], label='train')
        plt.plot(hist.history['val_loss'][50:], label='val', color = 'red')
        plt.legend()
        plt.show()

    else:
        model.compile(
                        loss=[ kwargs['loss_data'], kwargs['loss_data']],
                                loss_weights = [kwargs['w_mae'],kwargs['wph']],
                                optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                metrics=[
                                        "mean_absolute_error",  
                                        ])
        hist, model = model_fit_physics(model,
                                  extras_train,
                                  extras_test,
                                  intras_train,
                                  np.zeros(extras_train.shape) ,
                                  intras_test,
                                  np.zeros(extras_test.shape)  ,
                                    kwargs['epoch'], 
                                 kwargs['b_s'] )

    return model



def train_model_qt (data,train_sets, test_sets, **kwargs):
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    tf.keras.backend.clear_session()
    
    intras_test,extras_test =data_prep_test (data,test_sets)
    intras_train,extras_train =data_prep_test (data,train_sets)
    model = Unet_qt(input_shape=(lenghts,1), layer_n =kwargs['ch_num'] ,kernel_size = kwargs['kernel_size'], depth = kwargs['depth'], physics=kwargs['physics'],use_com=True )
    if kwargs['physics'] == False:
        model.compile(
                            loss=[ kwargs['loss_data']],
                                    optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                    metrics=[
                                            "mean_absolute_error",  
                                            ])
        hist, model = model_fit(model,
                                      extras_train,
                                      extras_test,
                                      intras_train,
                                      intras_test,
                                  kwargs['epoch'], 
                                 kwargs['b_s'] )       

        plt.plot(hist.history['loss'][50:], label='train')
        plt.plot(hist.history['val_loss'][50:], label='val', color = 'red')
        plt.legend()
        plt.show()

    else:
        model.compile(
                        loss=[ kwargs['loss_data_q1'],
                               kwargs['loss_data']],
                                loss_weights = [kwargs['w_mae'],
                                                kwargs['wph']],
                                optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                metrics=[
                                        "mean_absolute_error",  
                                        ])
        hist, model = model_fit_physics_qt(model,
                                  extras_train,
                                  extras_test,
                                  intras_train,
                                  np.zeros(extras_train.shape) ,
                                  intras_test,
                                  np.zeros(extras_test.shape)  ,
                                    kwargs['epoch'], 
                                 kwargs['b_s'] )

    return model



def train_model_qt_all (data,train_sets, test_sets, **kwargs):
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    tf.keras.backend.clear_session()
    
    intras_test,extras_test =data_prep_test (data,test_sets)
    intras_train,extras_train =data_prep_test (data,train_sets)
    model = Unet_qt_all(input_shape=(lenghts,1), layer_n =kwargs['ch_num'] ,kernel_size = kwargs['kernel_size'], depth = kwargs['depth'], physics=kwargs['physics'],use_com=True )
    if kwargs['physics'] == False:
        model.compile(
                            loss=[ kwargs['loss_data']],
                                    optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                    metrics=[
                                            "mean_absolute_error",  
                                            ])
        hist, model = model_fit(model,
                                      extras_train,
                                      extras_test,
                                      intras_train,
                                      intras_test,
                                  kwargs['epoch'], 
                                 kwargs['b_s'] )       

        plt.plot(hist.history['loss'][50:], label='train')
        plt.plot(hist.history['val_loss'][50:], label='val', color = 'red')
        plt.legend()
        plt.show()

    else:
        model.compile(
                        loss=[ kwargs['loss_data_q1'],kwargs['loss_data_q2'],kwargs['loss_data_q3'],
                               kwargs['loss_data'],kwargs['loss_data'],kwargs['loss_data']],
                                loss_weights = [kwargs['w_mae'],kwargs['w_mae'],kwargs['w_mae'],
                                                kwargs['wph'],kwargs['wph'],kwargs['wph'],
                                               ],
                                optimizer=tf.keras.optimizers.Adam(learning_rate = kwargs['lr']),
                                metrics=[
                                        "mean_absolute_error",  
                                        ])
        hist, model = model_fit_physics_qt_all(model,
                                  extras_train,
                                  extras_test,
                                  intras_train,
                                  np.zeros(extras_train.shape) ,
                                  intras_test,
                                  np.zeros(extras_test.shape)  ,
                                    kwargs['epoch'], 
                                 kwargs['b_s'] )

    return model


def quantile_loss(q):
    '''
    Returns the quantile loss for a given quantile `q`.
    `q` should be a float between 0 and 1. 
    '''
    def loss(y_true, y_pred):
        error = tf.subtract(y_true, y_pred)
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    
    return loss


def models_comparision(rmse_models,extras_test,intras_test,extras_train):
    df_results = pd.DataFrame()
    dic_results = {}

    dic_results_train = {}

    for counter,i in enumerate(rmse_models):
        print(i)
        name =i.split('/')[1].split('_')
        ph = name[1]
        seed = name[2].split('.')[0][-1]
        model = load_model(i, compile = False)
        if ph == 'True':
            print(ph)
            preds = model.predict(extras_test)[0].reshape(-1,8000)
            train_preds = model.predict(extras_train)[0].reshape(-1,8000)
            ph = True
        else:
            preds = model.predict(extras_test).reshape(-1,8000)
            print(preds.shape)
            
            train_preds = model.predict(extras_train).reshape(-1,8000)
            ph = False
        dic_results[i] = preds
        dic_results_train[i] =  train_preds
        df_results.loc[counter,'ph']=ph
        df_results.loc[counter,'seed']=seed
        df_results.loc[counter,'mae']=mean_absolute_error(intras_test, preds)
        df_results.loc[counter,'mse']=mean_squared_error(intras_test, preds, squared=True)
        df_results.loc[counter,'rmse']=mean_squared_error(intras_test, preds, squared=False)    

    return df_results,dic_results,dic_results_train


def model_performance (selected_model,intras_test,extras_test,extras_train,intras_train,apds_test,apds_train,data_feature,UNSEEN, quantile = False):
    APD_preds =['APD_Pred_'+str(10*i) for i in range(1,11)] 
    APD_vals = ['APD_Val_'+str(10*i) for i in range(1,11)] 

    APD_ER = ['APD_ER_'+str(10*i) for i in range(1,11)]
    APD_PER = ['APD_PER_'+str(10*i) for i in range(1,11)]

    data_feature_test=data_feature[data_feature['name']==UNSEEN][ ['ind','w1+w2', 'w3+w1+w2', 'w3', 'h1', 'h2', 'h1/h2', 'h3/h1', 'h3/h2', 'h3', 'r3', 'd3','APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60','APD70', 'APD80', 'APD90', 'APD100','s/n','iap_sn','rs/n' ]]
    print(data_feature_test.shape)
    name = selected_model.split('/')[1].split('.')[0]
    model = load_model(selected_model, compile = False)
    ph = selected_model.split('/')[1].split('_')[1]
    preds = model.predict(extras_test)[0].reshape(-1,8000)
    preds_train = model.predict(extras_train)[0].reshape(-1,8000)
    act_vs_pred_plot( intras_test.reshape(-1), preds.reshape(-1),name)
    apds_pred_test = get_all_apds(preds)
    apds_pred_train = get_all_apds(preds_train)
    apd_comp_plot(apds_pred_test,apds_test,apds_pred_train,apds_train,name)
    print(preds.shape,intras_test.shape,preds_train.shape,intras_train.shape)
    error  = np.abs(preds- intras_test).mean(axis = 1)
    
    error_train  = np.abs(preds_train- intras_train).mean(axis = 1)
    
    mae_error( error,error_train, name+'_MAE')
    error_apd  = np.abs(apds_pred_test- apds_test).sum(axis = 1)/5000
    error_train_apd  = np.abs(apds_pred_train- apds_train).sum(axis = 1)/5000
    mae_error( error_apd,error_train_apd, name+'_APD')

    apd_errors = np.abs(apds_pred_test- apds_test)
    apd_perrors = apd_errors*100/apds_test
    data_feature_test['mae'] = error
    data_feature_test['traingle'] = data_feature_test['APD30']/data_feature_test['APD90']
    data_feature_test[APD_ER] =apd_errors/5000
    data_feature_test[APD_PER] = apd_perrors
    data_feature_test['APD_ER_Mean'] =apd_errors.sum(axis = 1)/5000
    data_feature_test['APD_PER_Mean'] =apd_perrors.sum(axis = 1)
    col1=[ 'h1','w3','h3', 's/n', 'd3',  'traingle','rs/n' ]
    plot_errorss(data_feature_test,'mae',name,col1)
    return data_feature_test


def model_performance_quantile (selected_model,intras_test,extras_test,extras_train,intras_train,apds_test,apds_train,data_feature,UNSEEN):
    APD_preds =['APD_Pred_'+str(10*i) for i in range(1,11)] 
    APD_vals = ['APD_Val_'+str(10*i) for i in range(1,11)] 

    APD_ER = ['APD_ER_'+str(10*i) for i in range(1,11)]
    APD_PER = ['APD_PER_'+str(10*i) for i in range(1,11)]

    data_feature_test=data_feature[data_feature['name']==UNSEEN][ ['ind','w1+w2', 'w3+w1+w2', 'w3', 'h1', 'h2', 'h1/h2', 'h3/h1', 'h3/h2', 'h3', 'r3', 'd3','APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60','APD70', 'APD80', 'APD90', 'APD100','s/n','iap_sn','rs/n' ]]
    data_feature_train=data_feature[data_feature['name']!=UNSEEN][ ['ind','w1+w2', 'w3+w1+w2', 'w3', 'h1', 'h2', 'h1/h2', 'h3/h1', 'h3/h2', 'h3', 'r3', 'd3','APD10', 'APD20', 'APD30', 'APD40', 'APD50', 'APD60','APD70', 'APD80', 'APD90', 'APD100','s/n','iap_sn','rs/n' ]]

    print(data_feature_test.shape)
    name = selected_model.split('/')[1].split('.')[0]
    model = load_model(selected_model, compile = False)

    preds_med = model.predict(extras_test)[1].reshape(-1,8000)
    preds_l = model.predict(extras_test)[0].reshape(-1,8000)
    preds_u = model.predict(extras_test)[2].reshape(-1,8000)

    preds = preds_med
    print(intras_test.shape,preds_med.shape,preds_l.shape,preds_u.shape)
    preds_train = model.predict(extras_train)[1].reshape(-1,8000)
    act_vs_pred_plot_quantile( intras_test.reshape(-1),[preds_l.reshape(-1),preds_med.reshape(-1),preds_u.reshape(-1)],name)
    apds_pred_test = get_all_apds(preds_l)
    apds_pred_train = get_all_apds(preds_train)
    apd_comp_plot(apds_pred_test,apds_test,apds_pred_train,apds_train,name)
    print(preds.shape,intras_test.shape,preds_train.shape,intras_train.shape)
    error  = np.abs(preds- intras_test).mean(axis = 1)
    
    error_train  = np.abs(preds_train- intras_train).mean(axis = 1)
    
    mae_error( error,error_train, name+'_MAE')
    error_apd  = np.abs(apds_pred_test- apds_test).sum(axis = 1)/5000
    error_train_apd  = np.abs(apds_pred_train- apds_train).sum(axis = 1)/5000
    mae_error( error_apd,error_train_apd, name+'_APD')

    apd_errors = np.abs(apds_pred_test- apds_test)
    apd_perrors = apd_errors*100/apds_test

    apd_errors_train = np.abs(apds_pred_train- apds_train)
    apd_perrors_train = apd_errors_train*100/apds_train

    data_feature_test['mae'] = error
    data_feature_test['traingle'] = data_feature_test['APD30']/data_feature_test['APD90']
    data_feature_test[APD_ER] =apd_errors/5000
    data_feature_test[APD_PER] = apd_perrors
    data_feature_test['APD_ER_Mean'] =apd_errors.sum(axis = 1)/5000
    data_feature_test['APD_PER_Mean'] =apd_perrors.sum(axis = 1)

    data_feature_train['mae'] = error_train
    data_feature_train['traingle'] = data_feature_train['APD30']/data_feature_train['APD90']
    data_feature_train[APD_ER] =apd_errors_train/5000
    data_feature_train[APD_PER] = apd_perrors_train
    data_feature_train['APD_ER_Mean'] =apd_errors_train.sum(axis = 1)/5000
    data_feature_train['APD_PER_Mean'] =apd_perrors_train.sum(axis = 1)

    col1=[ 'h1','w3','h3', 's/n', 'd3',  'traingle','rs/n' ]
    plot_errorss(data_feature_test,'mae',name,col1)
    return data_feature_test,data_feature_train,error_apd,error_train_apd

def predictor_all(model1,list_of_data):
    preds = model1.predict(list_of_data)
    pred1=preds[0].reshape(-1,8000)
    pred2=preds[1].reshape(-1,8000)
    pred3=preds[2].reshape(-1,8000)
    return pred1,pred2,pred3


def multi_channel_pred (data_extras,model,times,fs,lowcut,highcut,ps, ds,order,baseline_ch = 0 ):

    dic_eaps = {}
    main_counter = 0
    df = pd.DataFrame()
    for t_couner, t in enumerate(times[1:]):
        #baseline here is used to detect peaks precsiley based on the channel which has the highest amplitude
        b0 = baseline(data_extras,0 ,  t, fs,lowcut,order,highcut,ps, ds )
        for i in range(len(data_extras)):
            print('channel',i,t)
            data_channel_raw = data_extras[i] [t[0]:t[1]] # time segment 
            data_channel_a,nois_channel_a,noise_level = screen_noise_noiselevel(data_channel_raw, lowcut,highcut, fs,order)
    
            for counter, p in enumerate(b0[2:-2]) :
                n_points = 850
                extra_segmented,extra_segmented_a,extra_noise_segmented_a = get_window_with_noise(data_channel_a,data_channel_raw,nois_channel_a,p,n_points= n_points)
                sp_0,npower_0,sn_0 ,width_extras_a = get_info(extra_segmented,extra_noise_segmented_a,noise_level)
                std_,span0,snr=  sn_ratio (extra_segmented)
                width_extras_raw = get_width(extra_segmented)
                df.loc[main_counter,['sp','np','sn_p','snr','span','ch','t','p','#','width_extras_a','width_extras_raw','amp','std_',]] = sp_0,npower_0,sn_0,snr,span0, int(i), int(t_couner),int(p+t[0]),int(counter),width_extras_a,width_extras_raw,np.max(extra_segmented_a)-np.mean(extra_segmented_a[0:100]),std_
    
                try:
                    eap = extra_segmented
                    eap_norm = function2(extra_segmented)
                    x1 = np.argmax(eap)
                    x2 = np.argmin(eap[x1:x1+100])+x1
                    y1 = np.max(eap_norm)
                    ry1 = np.max(eap)
                    y2 = eap_norm[x2]
                    ry2 = eap[x2]
                    x3,ry3 = eap_x3(eap,x1)
                    y3  = eap_norm[x3]
                    x_ch2, ry_ch2 =  eap_ch2_3(eap,x2)
                    y_ch2 = eap_norm[x_ch2]
                    ry_ch22= np.max([eap[x2:x2+300]])
                    y_ch22= np.max([eap_norm[x2:x2+300]])
                    x_ch22= np.argmax([eap[x2:x2+300]])+x2
                    decay = (y_ch2-y3)/(x3-x_ch2)
                    rdecay = (ry_ch2-ry3)/(x3-x_ch2)
                    decay_new = abs(y3/(x3-x1))
                    rdecay_new = abs(ry3/(x3-x1))
                    var,var_diff,rel_var = variation(eap)
                    var_norm,var_diff_norm,rel_var_norm = variation(eap_norm)
                    df.loc[main_counter,['x1','x2','x3','y1','y2','y3','ry1','ry2','ry3','x_ch2','y_ch2','ry_ch2','x_ch22','y_ch22','ry_ch22']]= x1,x2,x3,y1,y2,y3,ry1,ry2,ry3,x_ch2,y_ch2,ry_ch2,x_ch22,y_ch22,ry_ch22
                    df.loc[main_counter,['decay','rdecay','decay_new','rdecay_new']]=decay,rdecay,decay_new,rdecay_new
                    df.loc[main_counter,['y0']]=np.mean(eap_norm[:200])
                    df.loc[main_counter,['ry0']]=np.mean(eap[:200])
                    df.loc[main_counter,['var','var_norm']]=var,var_norm
                    df.loc[main_counter,['var_diff','var_diff_norm']]=var_diff,var_diff_norm
                    df.loc[main_counter,['rel_var','rel_var_norm']]=rel_var,rel_var_norm
                except:
                    pass
    
                dic_eaps[(main_counter,'eap_raw')] = extra_segmented
                dic_eaps[(main_counter,'eap_raw_norm')] = function2(extra_segmented)
                dic_eaps[(main_counter,'eap_a_norm')] = function2(extra_segmented_a)
                main_counter = main_counter+1
                
    df['y_ch2_rel']= df['y_ch2']-df['y0']
    df['ry_ch2_rel']= df['ry_ch2']-df['ry0'][:]
    df['y_ch2_rel2']= df['y_ch22']-df['y0']
    df['ry_ch2_rel2']= df['ry_ch22']-df['ry0'][:]
    df['y3/y1'] =df['ry_ch2_rel']/df['amp']
    df2= df.copy()
    df2= df[(df['x1']<1000) & (df['x2']<1000) ] #making sure the peaks are correctly found


    
    eaps = np.array([dic_eaps[(i,'eap_raw_norm')] for i in df2.index])
    pred1,pred2,pred3 =  predictor_all(model,eaps)
    preds1_sm = smoother(pred1)
    preds2_sm = smoother(pred2)
    preds3_sm = smoother(pred3)
    ci = np.abs(pred3[:, 1000:] - pred1[:, 1000:]).mean(axis=1)
    ci_smooth = np.abs(preds3_sm[:, 1000:] - preds1_sm[:, 1000:]).mean(axis=1)
    df2.loc[df2.index,'ci']=ci
    df2.loc[df2.index,'ci_smooth']=ci_smooth
    for p1,p2,p3,ps1,ps2,ps3, ind in zip (pred1,pred2,pred3,
                                             preds1_sm,preds2_sm,preds3_sm,
                                          df2.index):
            dic_eaps[(ind,'p1')]=p1
            dic_eaps[(ind,'p2')]=p2
            dic_eaps[(ind,'p3')]=p3
            dic_eaps[(ind,'p1_smooth')]=ps1
            dic_eaps[(ind,'p2_smooth')]=ps2
            dic_eaps[(ind,'p3_smooth')]=ps3

    return df2, dic_eaps

