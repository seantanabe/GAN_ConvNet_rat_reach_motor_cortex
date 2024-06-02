
#%% ConvNet P2: train dense, leave-one-out on 10 reach, plot top predicted neural activity

%reset -f

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import scipy.io as sio
# import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Activation, Input, Lambda, MaxPooling2D, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, Input, Conv2D, Reshape, RepeatVector, Permute, GRU, SimpleRNN, BatchNormalization
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from keras import backend as K

from sklearn.metrics import r2_score 

load_path = 'D:\\20220624_Lethbridge\\20230111_rat_unit_EckertTatsuno\\20240324_sample_code_dt\\rr5\\input_convnet_p2'
save_path = 'D:\\20220624_Lethbridge\\20230111_rat_unit_EckertTatsuno\\20240324_sample_code_dt\\rr5\\output_convnet_p2'

rr_day = [[5,7]] # [rr,day]

N_leaveOne = 10

grid_shift_iPR_vid = [0] #np.arange(-1000,1000+1,50,dtype=int) # ms
grid_iPRDur    = [1000] # ms
grid_iPRIncr   = [50] # ms
grid_VidDur    = [5] # ms
grid_VidIncr   = [15] # ms
grid_zoom      = [1]
N_lp = len(grid_shift_iPR_vid)*len(grid_iPRDur)*len(grid_iPRIncr)*len(grid_VidDur)*len(grid_VidIncr)*len(grid_zoom)
param_lp = np.zeros((N_lp,6)) 
count = 0
for i in grid_iPRDur:
    for ii in grid_iPRIncr:
        for iii in grid_VidDur:
            for iiii in grid_VidIncr:
                for iiiii in grid_shift_iPR_vid:
                    for iiiiii in grid_zoom:
                            param_lp[count,:] = [i, ii, iii, iiii, iiiii, iiiiii]
                            count += 1
                            
for rr_day_i in np.arange(len(rr_day)):
    # rr_day_i = 1
    rr_i = rr_day[rr_day_i][0] 
    day_i = rr_day[rr_day_i][1] 

    for vid_i in np.arange(N_lp):
        # vid_i = 0
        iPRDur    = int(param_lp[vid_i,0])
        iPRIncr   = int(param_lp[vid_i,1])
        VidDur    = int(param_lp[vid_i,2])
        VidIncr   = int(param_lp[vid_i,3])
        shift_i   = int(param_lp[vid_i,4])
        zoom_i    = int(param_lp[vid_i,5])
        
        if zoom_i == 1:
            zoom_str = 'zoom_'
        else:
            zoom_str = ''
    
        os.chdir(load_path)
        spk_count = sio.loadmat('rat' + str(rr_i) + '_day' + str(day_i) + '_iPR_incr_neuron_iPRDur' + str(iPRDur) + 'ms_iPRIncr' + str(iPRIncr) + 'ms.mat')['iPR_tr_stack_incr']
        
        N_reach      = np.size(spk_count,1)
        N_neuron_all = np.size(spk_count,0)
        N_incr       = np.size(spk_count,2)
        N_sample     = N_reach*N_incr
        
        for i in np.arange(N_reach):
            # i = 0
            print('load reach ' + str(i))
            if shift_i < 0:
                str_n = 'n';
            else:
                str_n = '';   
            tmp_fr = sio.loadmat('rat' + str(rr_i) + '_day' + str(day_i) + '_vid_' + zoom_str + 'incr_reach_' + str(i) + '_iPRDur' + str(iPRDur) + 'ms_iPRIncr' + str(iPRIncr) + 'ms_' + 
                        'shiftiPRvid' + str_n + str(abs(shift_i)) + 'ms_cnn.mat')['frame_reach_i']
            if i == 0:
                N_feature = np.size(tmp_fr,2)
                N_frame   = np.size(tmp_fr,1)
                vid_frames_all = np.zeros((N_reach, N_incr, N_frame, N_feature))  
            vid_frames_all[i,:,:,:] = tmp_fr
        
        
        reach_leaveOne = np.arange(0,N_reach, np.floor(N_reach/N_leaveOne))
        reach_leaveOne = reach_leaveOne[0:10].astype(int)
        groups_reach  = np.zeros((N_reach,1))
        count = 0
        for i_lo in np.arange(1,N_leaveOne+1):
            groups_reach[reach_leaveOne[i_lo-1],:] = i_lo

        os.chdir(save_path)
        np.save(('rat' + str(rr_i) + '_day' + str(day_i) + '_group_reach.npy'),groups_reach) 
        r2_neuron = np.zeros((N_neuron_all))
        for i_neur in np.arange(N_neuron_all):
            # i_neur = 3
            spk_count_i = np.expand_dims(spk_count[i_neur,:,:], axis = 0)
            N_neuron = np.size(spk_count_i,0)
             
            vid_frames_sample = np.zeros((N_sample, N_frame, N_feature))  
            spk_count_sample  = np.zeros((N_sample, N_neuron))  
            groups_sample     = np.zeros((N_sample,1))  
            for i in np.arange(N_reach):
                print('rr ' + str(rr_i) + ' day ' + str(day_i) + ' prepro reach ' + str(i))
                tmp_ind = np.arange((i*N_incr),(i*N_incr+N_incr))
                groups_sample[tmp_ind,0] = np.full(N_incr, groups_reach[i])
                vid_frames_sample[tmp_ind,:,:] = vid_frames_all[i,:,:,:]
                spk_count_sample[tmp_ind,:] = spk_count_i[:,i,:].T
            
            X = vid_frames_sample
            y = spk_count_sample
            del vid_frames_sample, spk_count_sample
            
            def coeff_det(y_true, y_pred):
                SS_res =  K.sum(K.square( y_true-y_pred ))
                SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
                return ( 1 - SS_res/(SS_tot + K.epsilon()) )
            
            def coeff_det_neurons(y_true, y_pred):
                cd_neur = np.zeros(N_neuron)
                for i in np.arange(N_neuron):
                    # i = 0
                    SS_res =  K.sum(K.square( y_true[:,i]-y_pred[:,i] ))
                    SS_tot = K.sum(K.square( y_true[:,i] - K.mean(K.constant(y_true[:,i]) ) ))
                    cd_neur[i] = ( 1 - tf.cast(SS_res, tf.float32)/(SS_tot + K.epsilon()) )
                return cd_neur
            
            class CustomCallback(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    print("rr {} day {} loop {} of {} fold {} neuron {} shift {} iPRDur {} iPRIncr {} VidDur {} VidIncr {} zoom {}".format(rr_i, day_i, vid_i, N_lp, group_i, i_neur, shift_i, iPRDur, iPRIncr, VidDur, VidIncr, zoom_i))
                def on_epoch_end(self, epoch, logs=None):
                    y_pred = model.predict(X_test) 
                    r2_epoch_neurons[:,epoch] = coeff_det_neurons(y_test, y_pred)
                    
            
           
            for group_i in np.arange(1,N_leaveOne+1):
                # group_i = 2
                test_index  = np.where(groups_sample == group_i)[0]
                train_index = np.where(groups_sample != group_i)[0]
                X_train, X_test = X[train_index,:,:], X[test_index,:,:]
                y_train, y_test = y[train_index,:],     y[test_index,:]
                
                backend.clear_session()
                model = Sequential()
                
                # model.add(Dense(128, activation='relu'))
                # model.add(Dense(16, activation='relu'))
                model.add(Dense(N_neuron)) 
                model.add(Activation('sigmoid'))
                optimizers.Adam(learning_rate=0.001) #01)
                model.compile(loss='MSE',optimizer='Adam', metrics=[coeff_det]) 
                # model.summary()
                
                epochs = 400
                batch_size= 128 # 128 # 256 #512 #1024
                r2_epoch_neurons = np.zeros((N_neuron,epochs))
                history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[CustomCallback()]) # batch_size=1024, callbacks=[CustomCallback()])
                y_pred = model.predict(X_test) 
                
                
                if group_i == 1:
                    y_pred_lo = y_pred
                    y_test_lo = y_test
                else:
                    y_pred_lo = np.concatenate([y_pred_lo, y_pred])
                    y_test_lo = np.concatenate([y_test_lo, y_test])
                
                history_cat = [np.asarray(history.history['loss']), 
                       np.asarray(history.history['coeff_det']),
                       np.asarray(history.history['val_loss']),
                       np.asarray(history.history['val_coeff_det'])]
                
                # plt.figure(figsize=(2.7,2.5))
                # plt.plot(history.history['val_coeff_det'])
                # plt.ylim(0, 1) 
                # plt.xlim(0, epochs) 
                # plt.grid()
                
                if shift_i < 0:
                    str_n = 'n';
                else:
                    str_n = '';   
                    
                save_suffix = ('rr' + str(rr_i) + '_day' + str(day_i) + '_neur' + str(i_neur) 
                           + '_shift' + str_n + str(np.abs(shift_i)) + 'ms'
                           + '_iPRDur' + str(iPRDur) + 'ms_iPRIncr' + str(iPRIncr) + 'ms_VidDur' + str(VidDur) 
                           + 'ms_VidIncr' + str(VidIncr) + 'ms_fold' + str(group_i) + '_zoom' + str(zoom_i) + '_group' + str(group_i))
                np.save((save_path + '\\predict_' + save_suffix + '.npy'),y_pred)
                model.save(save_path + '\\model_' + save_suffix + '.h5') #, save_format='h5')

            r2_neuron[i_neur] = r2_score(np.squeeze(y_test_lo), np.squeeze(y_pred_lo))
            np.save((save_path + '\\' + 'neur' + str(i_neur)  + '_y_test_ol.npy'),y_test_lo)
            np.save((save_path + '\\' + 'neur' + str(i_neur)  + '_y_pred_ol.npy'),y_pred_lo)

np.save((save_path + '\\r2_neuron.npy'),r2_neuron)    

neuron_ind_top6 = np.flip(np.argsort(r2_neuron)[-6:])

###
### plotting 1 reach's neural activity prediction of best predicted neuron
t_vec = np.arange(-400,600+1,50)
i_neur = neuron_ind_top6[0]
y_test_lo = np.load(save_path + '\\' + 'neur' + str(i_neur)  + '_y_test_ol.npy')
y_pred_lo = np.load(save_path + '\\' + 'neur' + str(i_neur)  + '_y_pred_ol.npy')
reach_i = 2
plt.plot(t_vec,np.squeeze(y_test_lo)[reach_i*21:(reach_i+1)*21], color="black")
plt.plot(t_vec,np.squeeze(y_pred_lo)[reach_i*21:(reach_i+1)*21], color="red")
plt.ylabel("neural activity")
plt.xlabel("time from reach (ms)")
plt.margins(x=0)
plt.tight_layout() 

###
### plotting 10 leave-one out reach, for top 6 predicted neurons
count_i = 0
for i_neur in neuron_ind_top6:
    count_i += 1
    axes = plt.subplot(3, 2, count_i)
    y_test_lo = np.load(save_path + '\\' + 'neur' + str(i_neur)  + '_y_test_ol.npy')
    y_pred_lo = np.load(save_path + '\\' + 'neur' + str(i_neur)  + '_y_pred_ol.npy')
    plt.plot(np.squeeze(y_test_lo), color="black")
    plt.plot(np.squeeze(y_pred_lo), color="red")
    plt.margins(x=0)
    plt.tight_layout() 
    axes.title.set_text('$R^{2}$ =' + str(round(r2_neuron[i_neur],2)) + ' (neuron ' + str(i_neur) + ')' )
    if count_i == 5:
        axes.set_ylabel("neural activity")
        axes.set_xlabel("time unit")
        

