import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Reshape, Conv1D,Conv2D, BatchNormalization, Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply, Flatten, LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from keras.models import load_model
import random
from utils_q import *


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


def gaussian_kernel(size: int, std: float):
    x = np.linspace(-size // 2 + 1, size // 2, size)
    kernel = np.exp(-x**2 / (2 * std**2))
    kernel = kernel / np.sum(kernel)
    return kernel

# Create a smoothing function for tensors with shape (batchsize, 8000, 1)
def smooth_predictions(predictions, kernel_size=7001, std=30.0):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    
    # Create the Gaussian kernel
    gaussian_filter = gaussian_kernel(kernel_size, std).astype(np.float32)
    gaussian_filter = gaussian_filter[:, np.newaxis, np.newaxis]
    
    # Apply the convolution across the batch
    smoothed_predictions = tf.nn.conv1d(predictions, gaussian_filter, stride=1, padding='SAME')
    
    return smoothed_predictions


def physics_compactor1(x,kernel_size):
            bk =  cbr(x, 32*2, kernel_size, 5, 1)
            print('bk-noncom1',bk.shape)
            bk =  cbr(bk, 32*3, kernel_size, 5, 1)
            print('bk-noncom2',bk.shape)
            bk =  cbr(bk, 32*4, kernel_size, 5, 1)
            print('bk-noncom3',bk.shape)
            bk = tf.keras.layers.GlobalAveragePooling1D()(bk)
            print('bk-noncom4',bk.shape)
            return bk

def dense_to_physics(bk):
        b = tf.keras.layers.Flatten()(bk)
        b = tf.keras.layers.BatchNormalization()(b)
        print('b',b.shape)        
        b= tf.keras.layers.Dense(8,kernel_initializer= tf.keras.initializers.HeNormal())(b )
        b=  tf.keras.layers.Activation("relu")(b)
        # b= tf.keras.layers.Dropout(0.2)(b)
        print('b_1',b.shape)  
        return b

def physics_part(input_layer,v_out,b,use_cond,name):
       
        # t_out = tf.keras.layers.Dense(1)(b)
        # t_out = tf.keras.layers.Activation("relu")(t_out)
        # t_out = tf.clip_by_value(t_out, clip_value_min=0.05, clip_value_max=0.45)
        # t_out = t_out*8
        # t_out = tf.expand_dims(t_out, axis=1)
        t_out = tf.ones_like(v_out)*1.6
        t_ = t_out
    
        k_out = tf.keras.layers.Dense(1)(b)
        k_out = tf.keras.layers.Activation("relu")(k_out)
        k_out = tf.clip_by_value(k_out, clip_value_min=0.1, clip_value_max=1)
        k = k_out

        a_out = tf.keras.layers.Dense(1)(b)
        a_out = tf.keras.layers.Activation("relu")(a_out)
        a_out = tf.clip_by_value(k_out, clip_value_min=0.001, clip_value_max=0.1)
        a = a_out

        a_out = tf.expand_dims(a_out, axis=1) 
        k_out = tf.expand_dims(k_out, axis=1)
        print('a', a_out.shape)
        print('k', k_out.shape)
        
        x_out = tf.keras.layers.Dense(1)(b)
        x_out = tf.keras.layers.Activation("relu")(x_out)
        x_out = tf.clip_by_value(x_out, clip_value_min=0.1, clip_value_max=0.6)
        x_out = 1 - x_out
        x = x_out
        x_out = tf.expand_dims(x_out, axis=1) 
        print('x', tf.shape(x_out))

        a_out = tf.broadcast_to(a_out , [tf.keras.backend.shape(input_layer )[0], tf.keras.backend.shape(input_layer )[1],1] )
        k_out = tf.broadcast_to(k_out , [tf.keras.backend.shape(input_layer )[0], tf.keras.backend.shape(input_layer )[1],1] )
        x_out = tf.broadcast_to(x_out , [tf.keras.backend.shape(input_layer )[0], tf.keras.backend.shape(input_layer )[1],1] )
        t_out = tf.broadcast_to(t_out , [tf.keras.backend.shape(input_layer )[0], tf.keras.backend.shape(input_layer )[1],1] )
        
        ones = tf.ones_like(v_out)
        # t_out = tf.ones_like(v_out) * 1.6
        v_out_smooth  = smooth_predictions(v_out)
        print('v_out2', v_out_smooth.shape)
    
        dv_l = tf.keras.layers.ZeroPadding1D(padding=(1, 0), name='dv_l' + name)(v_out_smooth)
        dv_r = tf.keras.layers.ZeroPadding1D(padding=(0, 1), name='dv_r' + name)(v_out_smooth)
        dv_t = (dv_r[:, 1:, :] - dv_l[:, :-1, :]) / (2 * (t_out / 7999)) 
        dv2_t = (dv_r[:, 1:, :] + dv_l[:, :-1, :] - 2 * v_out_smooth) / ((t_out / 7999)**2)
        print('dv_t'+name, dv_t.shape)
        print('dv2_t'+name, dv2_t.shape)
        print('ones'+name, ones.shape)
        
        a_v =  tf.keras.layers.Multiply(name = 'a_v'+name)([a_out,v_out_smooth])

        epsilon = 1e-3
        v_out_noisy = tf.where(tf.equal(v_out_smooth, 0), tf.ones_like(v_out_smooth) * epsilon, v_out_smooth)
        inv_v = v_out_noisy ** -1
        inv_v = tf.where(tf.math.is_nan(inv_v), tf.fill(tf.shape(inv_v), 0.0), inv_v)
    
        print('inv_v', inv_v.shape, inv_v )
        print('a_v', a_v.shape, a_v)
        term_l1 = tf.keras.layers.Multiply(name ='tl1'+name)([v_out_smooth,k_out ,  dv_t, ones - 2*v_out_smooth+a_out]) # * u
        print('term_l1', term_l1)
        term_l2 = tf.keras.layers.Multiply(name='tl2'+name)([inv_v , dv_t**2]) #* u
        print('term_l2', term_l2)
        term_l3 = dv2_t
        # term_l2 = tf.keras.layers.Multiply(name='tl2'+name)([inv_v2 , dv_t**2])
        # term_l3 = tf.keras.layers.Multiply(name='tl3'+name)([inv_v , dv2_t])
        print('term_l3', term_l3)
        term_lt = 1000* term_l1 + term_l2 - term_l3
        
        print('term_l1', term_l1.shape)
        print('term_l2', term_l2.shape)
        print('term_l3', term_l3.shape)
        print(term_lt.shape)

        step = x_out * (tf.keras.activations.sigmoid(1000 * (v_out_smooth - a_out))) +(1-x_out) * (tf.keras.activations.sigmoid(1000 * (-v_out_smooth + a_out)))
        term_r1 = tf.keras.layers.Multiply(name='tr1'+name)([v_out_smooth,k_out, v_out_smooth ** 2 + a_out - a_v])
        term_r2 = dv_t
        term_rt = tf.keras.layers.Multiply(name='trt'+name)([step, 1000*term_r1 + term_r2 ])
        dv_out = term_lt -term_rt
        if use_cond == True:
             print('cond-True')
             dv_out = tf.where(v_out <=0, tf.zeros_like(dv_out), dv_out)
                
        dv_out = tf.concat([dv_out[:,100 :950], dv_out[:, 1050:-100]], axis=1)
        epssss = 1e-5
        dv_out = tf.math.log(dv_out**2+epssss)
        return dv_out,dv_t,dv2_t,a,k*1000,x,t_,dv_l,dv_r,t_out


def Unet(input_shape=(8000,1), layer_n =64,kernel_size = 7, depth = 4, physics = True,use_se = True, use_com = False, use_cond = False, quantile = False ): #att_n
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
    print('cbr-4:', x.shape) 
    print('upS2 with',us,":",x.shape)
    out_1_at = att(x,out_1)
    print('out_1_at', out_1_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_1_at])
    print('d5', x.shape)
    x = cbr(x, layer_n*2, kernel_size, 1, 1)
    print('cbr-3:',x.shape) 
    print('upS3 with',us,":",x.shape)
    out_0_at = att(x,out_0)
    print('pre_co_-2', x.shape ,out_0_at.shape)
    x = tf.keras.layers.Concatenate()([ tf.keras.layers.UpSampling1D(us)(x), out_0_at])
    x = cbr(x, layer_n, kernel_size, 1, 1)    
    print('aft_con-1',x.shape)
    if quantile:
        v1 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
        v_out_q1 = tf.keras.layers.Activation("tanh", name = 'vq1')(v1)
        v2 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
        v_out_q2 = tf.keras.layers.Activation("tanh", name = 'vq2')(v2)
        v3 = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
        v_out_q3 = tf.keras.layers.Activation("tanh", name = 'vq3')(v3)
        if physics == True: #  physics with quantile
            print('ph')
            if use_com==True:
                print('Comp_ON')   
                print('before-comp',x_comp.shape)
                bk1 = cbr(x_comp, 1, kernel_size, 1, 1)
                bk2 = cbr(x_comp, 1, kernel_size, 1, 1)
                bk3 = cbr(x_comp, 1, kernel_size, 1, 1)
            else:
                bk = physics_compactor1(x,kernel_size)
            b = dense_to_physics(bk)
            dv_out_max1,dv_t1,dv2_t1,a_out1,k_out1,x_out1,t_out1,dv_l1,dv_r1,t_out_broad1 = physics_part(input_layer,v_out_q1,b,use_cond,'noq1')
            dv_out_max2,dv_t2,dv2_t2,a_out2,k_out2,x_out2,t_out2,dv_l2,dv_r2,t_out_broad2 = physics_part(input_layer,v_out_q2,b,use_cond,'noq2')
            dv_out_max3,dv_t3,dv2_t3,a_out3,k_out3,x_out3,t_out3,dv_l3,dv_r3,t_out_broad3 = physics_part(input_layer,v_out_q3,b,use_cond,'noq3')
            
            model =  tf.keras.Model(input_layer, [v_out_q1,v_out_q2,v_out_q3 ,
                                                  dv_out_max1,dv_out_max2,dv_out_max3,
                                                  dv_t1,dv_t2,dv_t3,
                                                  dv2_t1,dv2_t2,dv2_t3,
                                                  a_out1,a_out2,a_out3,
                                                  k_out1,k_out2,k_out3,
                                                  x_out1,x_out2,x_out3,
                                                  t_out1,t_out2,t_out3,
                                                  dv_l1,dv_l2,dv_l3,
                                                  dv_r1,dv_r2,dv_r3,
                                                  t_out_broad1,t_out_broad2,t_out_broad3,
                                                 ])        
        else: # no physics with quantile
            model =  tf.keras.Model(input_layer, [v_out_q1,v_out_q2,v_out_q3 ])
    else:#no quantile
        print('no_quantile')
        v = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(x)
        v_out = tf.keras.layers.Activation("tanh", name = 'v')(v)
        print(v_out.shape)
        if physics == True:
            print('ph')
            if use_com==True:
                print('Comp_ON')   
                print('before-comp',x_comp.shape)
                bk = cbr(x_comp, 1, kernel_size, 1, 1)
            else:
                bk = physics_compactor1(x,kernel_size)
            b = dense_to_physics(bk)
            dv_out_max,dv_t,dv2_t,a_out,k_out,x_out,t_out,dv_l,dv_r,t_out_broad = physics_part(input_layer,v_out,b,use_cond,'noq')
            print('v_out, dv_out_max,dv_t,dv2_t,a_out,k_out,x_out,t_out', v_out.shape, dv_out_max.shape,
                  dv_t.shape,dv2_t.shape,a_out.shape,k_out.shape,x_out.shape,t_out.shape)
            model =  tf.keras.Model(input_layer, [v_out, dv_out_max,dv_t,dv2_t,a_out,k_out,x_out,t_out,dv_l,dv_r,t_out_broad])
        else:
            model =  tf.keras.Model(input_layer, v_out)
    return model 
    
    
def scheduler(epoch, lr):
  if epoch < 50:
        return lr
  else:
        return lr * tf.math.exp(-0.05)

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
## no physics

def model_fit(model, 
              train_input1,
              val_input1,
              train_target1,
              val_target1,
              n_epoch, batch_size=32, quantile=False):
        
    callbacks_list = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    ]
    
    if quantile:
        history = model.fit(x=train_input1, 
                            y=[train_target1, train_target1, train_target1],
                            validation_data=(val_input1, [val_target1, val_target1, val_target1]),
                            batch_size=batch_size,
                            epochs=n_epoch,
                            shuffle=True, 
                            verbose=2,
                            callbacks=callbacks_list)
    else:
        history = model.fit(x=train_input1, 
                            y=train_target1,
                            validation_data=(val_input1, val_target1),
                            batch_size=batch_size,
                            epochs=n_epoch,
                            shuffle=True, 
                            verbose=2,
                            callbacks=callbacks_list)
    
    # Plot loss history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()

    return history, model



#with physics
def model_fit_physics(model, 
                      train_input1,
                      val_input1,
                      train_target1,
                      val_target1,
                      n_epoch, batch_size=32, quantile = False):
    
    callbacks_list = [NaNHandlingCallback(scheduler)]
    # Create dummy data for the additional outputs
    dummy_train_ode = tf.zeros((train_target1.shape[0], 7700, 1))  
    dummy_val_ode = tf.zeros((val_target1.shape[0], 7700, 1))     
    dummy_train_dv = tf.zeros((train_target1.shape[0], 8000, 1)) 
    dummy_train_dvr = tf.zeros((train_target1.shape[0], 8001, 1)) 
    dummy_val_dv = tf.zeros((val_target1.shape[0], 8000, 1))    
    dummy_val_dvr= tf.zeros((val_target1.shape[0], 8001, 1))    
    dummy_train_sc = tf.zeros((train_target1.shape[0], 1))  # Adjusted shape to (batch_size, 1)
    dummy_val_sc = tf.zeros((val_target1.shape[0], 1))      # Adjusted shape to (batch_size, 1)

    print("train_target1 shape:", train_target1.shape)
    print("dummy_train_ode shape:", dummy_train_ode.shape)
    print("dummy_train_dv shape:", dummy_train_dv.shape)
    print("dummy_train_sc shape:", dummy_train_sc.shape)
    
    print("val_target1 shape:", val_target1.shape)
    print("dummy_val_ode shape:", dummy_val_ode.shape)
    print("dummy_val_dv shape:", dummy_val_dv.shape)
    print("dummy_val_sc shape:", dummy_val_sc.shape)


    # [v_out, dv_out, dv_t, dv2_t, a, k, x, t]
    # dv_out,dv_t,dv2_t,a,k*1000,x,t_,dv_l,dv_r,t_out
    if quantile:
          history = model.fit(x=train_input1, y=[train_target1,train_target1,train_target1, 
                                           dummy_train_ode,dummy_train_ode,dummy_train_ode,
                                           dummy_train_dv,dummy_train_dv,dummy_train_dv,
                                           dummy_train_dv,dummy_train_dv,dummy_train_dv,
                                           dummy_train_sc,dummy_train_sc,dummy_train_sc,
                                           dummy_train_sc,dummy_train_sc,dummy_train_sc,
                                           dummy_train_sc,dummy_train_sc,dummy_train_sc,
                                           dummy_train_sc,dummy_train_sc,dummy_train_sc,
                                           dummy_train_dvr,dummy_train_dvr,dummy_train_dvr,
                                            dummy_train_dvr,dummy_train_dvr,dummy_train_dvr,
                                            dummy_train_dv,dummy_train_dv,dummy_train_dv,
                                          ],
                              validation_data=(val_input1, [val_target1,val_target1,val_target1,
                                                            dummy_val_ode,dummy_val_ode,dummy_val_ode,
                                                            dummy_val_dv,dummy_val_dv,dummy_val_dv,
                                                            dummy_val_dv,dummy_val_dv,dummy_val_dv,
                                                            dummy_val_sc,dummy_val_sc,dummy_val_sc,
                                                            dummy_val_sc,dummy_val_sc,dummy_val_sc,
                                                            dummy_val_sc,dummy_val_sc,dummy_val_sc,
                                                            dummy_val_sc,dummy_val_sc,dummy_val_sc,
                                                            dummy_val_dvr,dummy_val_dvr,dummy_val_dvr,
                                                            dummy_val_dvr,dummy_val_dvr,dummy_val_dvr,
                                                            dummy_val_dv,dummy_val_dv,dummy_val_dv,
                                                           ]),
                              batch_size=batch_size,
                              epochs=n_epoch,
                              shuffle=True, verbose=2,
                              callbacks=callbacks_list)

    else:
        
            history = model.fit(x=train_input1, y=[train_target1, 
                                                   dummy_train_ode,
                                                   dummy_train_dv,
                                                   dummy_train_dv,
                                                   dummy_train_sc,
                                                   dummy_train_sc,
                                                   dummy_train_sc,
                                                   dummy_train_sc,
                                                   dummy_train_dvr,dummy_train_dvr,dummy_train_dv,
                                                  ],
                                      validation_data=(val_input1, [val_target1,
                                                                    dummy_val_ode,
                                                                    dummy_val_dv,
                                                                    dummy_val_dv,
                                                                    dummy_val_sc,
                                                                    dummy_val_sc,
                                                                    dummy_val_sc,
                                                                    dummy_val_sc,
                                                                    dummy_val_dvr,dummy_val_dvr,dummy_val_dv
                                                                   ]),
                                      batch_size=batch_size,
                                      epochs=n_epoch,
                                      shuffle=True, verbose=2,
                                      callbacks=callbacks_list)
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()
    return history, model

def train_model(data,trains_with_limits, max_samples, train_sets, test_sets, **kwargs):
     
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    tf.keras.backend.clear_session()
    
    intras_test, extras_test = data_prep_test(data, test_sets,raw = False,  max_samples=100000, limit_keys=trains_with_limits)
    intras_train, extras_train = data_prep_test(data, train_sets,raw = False,  max_samples=max_samples, limit_keys=trains_with_limits)
    print(intras_test.shape, extras_test.shape)
    print(intras_train.shape, extras_train.shape)
    length = 8000  # Assuming length is derived from the input shape
    quan = kwargs['quantile'] 

    model = Unet(input_shape=(length, 1), 
                 layer_n=kwargs['ch_num'], 
                 kernel_size=kwargs['kernel_size'], 
                 depth=kwargs['depth'], 
                 physics=kwargs['physics'],
                 use_com=kwargs['use_com'],
                use_cond=kwargs['use_cond'],
                quantile=quan)

    
    if kwargs['physics'] == False:
        if quan:
            loss_ = [kwargs['loss_data_q1'],['loss_data_q2'],['loss_data_q3']]
        else:
            loss_ = [kwargs['loss_data']]
        model.compile(
            loss=loss_,
            optimizer=tf.keras.optimizers.Adam(learning_rate=kwargs['lr'], clipnorm=2.0),
            metrics=["mean_absolute_error"]
        )
        hist, model = model_fit(
            model,
            extras_train,
            extras_test,
            intras_train,
            intras_test,
            kwargs['epoch'], 
            kwargs['b_s'],
            quantile = kwargs['quantile']

        )
        plt.plot(hist.history['loss'][50:], label='train')
        plt.plot(hist.history['val_loss'][50:], label='val', color='red')
        plt.legend()
        plt.show()
    else:
        if quan:
             loss_  = [kwargs['loss_data_q1'],kwargs['loss_data_q2'],kwargs['loss_data_q3']] + [kwargs['loss_data']]*30
             loss_weights_ = [kwargs['w_mae']]*3 +    [kwargs['wph']]*3+ [0.0]*27
        else:
            loss_=[kwargs['loss_data'], kwargs['loss_data'], kwargs['loss_data'], kwargs['loss_data'], 
                  kwargs['loss_data'], kwargs['loss_data'], kwargs['loss_data'], kwargs['loss_data'] 
                 , kwargs['loss_data'], kwargs['loss_data'], kwargs['loss_data'] ],
            loss_weights_=[kwargs['w_mae'], kwargs['wph'], 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        model.compile(
            loss=loss_,
            loss_weights=loss_weights_,
            optimizer=tf.keras.optimizers.Adam(learning_rate=kwargs['lr'] , clipnorm=2.0),
            metrics=["mean_absolute_error"]
        )
        hist, model = model_fit_physics(
            model,
            extras_train,
            extras_test,
            intras_train,
            intras_test,
            kwargs['epoch'], 
            kwargs['b_s'],
            quantile = kwargs['quantile']

        )
                                
        plt.plot(hist.history['loss'][50:], label='train')
        plt.plot(hist.history['val_loss'][50:], label='val', color='red')
        plt.legend()
        plt.show()

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
def train_and_save_model_quantile(hyperparams,  loc, data, trains, tests, extras_test, intras_test,trains_with_limits, max_samples):
    model = train_model(data,trains_with_limits, max_samples, trains, tests, **hyperparams)
    new_dic = {}
    for i in [ 'seed', 'kernel_size', 'ch_num', 'depth','wph', 'lr', 'epoch', 'physics', 'max_sample', ]:
        new_dic[i] = hyperparams[i]
    name2_ = name_from_dic(new_dic, 4)
    model.save(loc + name2_)
    return model

