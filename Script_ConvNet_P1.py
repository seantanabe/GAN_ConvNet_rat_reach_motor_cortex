
#%% ConvNet P1: V3 CNN

%reset -f

import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.io as sio
import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# change working directory to folder "rr5" 
load_path = 'input_convnet_p1'
save_path = 'output_convnet_p1'

rr_day = [[5,7]] # [rr,day]

frame_diff = 0 # # frames

for rr_day_i in np.arange(len(rr_day)):
    # rr_day_i = 0
    rr_i = rr_day[rr_day_i][0] 
    day_i = rr_day[rr_day_i][1] 

    os.chdir(load_path)
    tr_adv_ms = np.load('day' + str(day_i) + '_adv_ms_noHandPin.npy')
    N_reach   = np.size(tr_adv_ms)
    
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    #for i in range(0,N_reach):
    for i in [0]: #range(0,N_reach):
        # i = 0
        print('rr ' + str(rr_i) + ' day ' + str(day_i) + ' reach ' + str(i))

        
        frame_reach_vid = np.load("day" + str(day_i) + "_frame_reach_vid_zoom_reach" + str(i) + ".npy")  
        if frame_diff > 0:
            frame_reach_vid = frame_reach_vid[:,:,frame_diff:] - frame_reach_vid[:,:,:-frame_diff]
            frame_reach_vid = cv2.normalize(frame_reach_vid, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            frame_reach_vid = frame_reach_vid.astype(np.uint16)
        # plt.figure(); plt.imshow(frame_reach_vid[:,:,300], cmap='gray', vmin=100, vmax=200); plt.show()
            
        frame_reach_vid_pre = np.transpose(np.tile(frame_reach_vid, (3,1,1,1)), (3, 1, 2, 0)) 
        frame_reach_vid_prepro = preprocess_input(frame_reach_vid_pre)
        frame_reach_vid_prepro_i = np.transpose(frame_reach_vid_prepro[:,:,:,0], (1, 2, 0)) 
        # frame_vid_i_plt = frame_reach_vid_prepro[0,:,:,0]
        # plt.imshow(frame_vid_i_plt, cmap='gray')
        features_vid = base_model.predict(frame_reach_vid_prepro)
        os.chdir(save_path)
        np.save(("day" + str(day_i) + "_frame_reach_vid_zoom_reach" + str(i) + "_prepro.npy"),frame_reach_vid_prepro_i)    
        np.save(("day" + str(day_i) + "_frame_reach_vid_zoom_reach" + str(i) + "_cnn.npy"),features_vid)    
        
 