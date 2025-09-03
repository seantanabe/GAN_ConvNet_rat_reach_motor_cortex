
#%% GAN P1: conditional GAN and prediction

%reset -f

from skimage.measure import block_reduce
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

from matplotlib import pyplot as plt
import scipy.io as sio
import os
import gc
import time

from tensorflow.keras.models import load_model
import scipy.ndimage as ndimage
    
# change working directory to folder "rr5" 
load_path = 'input_gan_p1_p2'
save_path = 'output_gan_p1'


rr_day = [[5,7]] # [rr,day]

t_ms_incr = 20 # ms
t_ms_seg = [-400, 600]

Conv_unit = 128 

os.chdir(load_path)
r2_all = sio.loadmat('r2_neurons_all_6N.mat')['r2_allall']
for rr_day_i in np.arange(len(rr_day)):
    # rr_day_i = 0
    rr_i = rr_day[rr_day_i][0] 
    day_i = rr_day[rr_day_i][1] 
    ind_tmp = np.where((r2_all[3,:] == rr_i) & (r2_all[4,:] == day_i))
    
    str_save_prefix = ('rat' + str(rr_i) + '_day' + str(day_i) + '_reach_iPR_conditional_generator')

    r2top_neur = r2_all[2,ind_tmp]
    r2top_neur[0][0] = r2top_neur[0][0][0:6]
    
    os.chdir(load_path)
    iPR = sio.loadmat('rat' + str(rr_i) + '_day' + str(day_i) + '_iPR_incr_neuron_iPRDur1000ms_iPRIncr' + str(t_ms_incr) + 'ms.mat')
    iPR_n = iPR["iPR_tr_stack_incr"]
    iPR_n = np.squeeze(iPR_n[r2top_neur[0][0]-1,:,:])
    N_iPR   = np.size(iPR_n,2)
    N_reach = np.size(iPR_n,1)
    N_neur  = np.size(iPR_n,0)
    
    t_frame = np.load('rat' + str(rr_i) + '_day' + str(day_i) + '_frame_reach_unit.npy')
    t_fpms = np.load('rat' + str(rr_i) + '_day' + str(day_i) + '_frame_reach_fpms.npy')
    ms_per_frame = round(1/t_fpms)
    t_ms = t_frame*ms_per_frame 
    
    
    count = 0
    iPR_ms  = zeros((N_neur,N_reach*N_iPR))
    for i_reach in np.arange(N_reach):
        # i_reach = 0
        print('load reach ' + str(i_reach))
        
        iPR_ms[:,((i_reach)*N_iPR):((i_reach+1)*N_iPR)] = np.squeeze(iPR_n[:,i_reach,:])
        
        vid_reach = np.load('rat' + str(rr_i) + '_day' + str(day_i) + '_frame_reach_vid_zoom_reach' + str(i_reach) + '_prepro.npy')
        N_pix_tmp = 128 
        shift_pix_x = 10
        shift_pix_y = 20
        vid_reach = vid_reach[shift_pix_y:(shift_pix_y+N_pix_tmp),shift_pix_x:(shift_pix_x+N_pix_tmp),:]
  
        N_pix = 128 #64
        
        
        tmp_abs = abs(t_ms - t_ms_seg[0])
        ind_start = np.where(tmp_abs == min(tmp_abs))[0]
        tmp_abs = abs(t_ms - t_ms_seg[1])
        ind_end   = np.where(tmp_abs == min(tmp_abs))[0]
        
        ts_frame = np.arange(ind_start, ind_end+1, t_ms_incr/ms_per_frame,dtype=int)
        for i_ind in np.arange(len(ts_frame)):
            if i_reach == 0 and i_ind == 0:
                ts_ms  = zeros(len(ts_frame)*(N_reach))
                ts_vid = zeros((len(ts_frame)*(N_reach),N_pix,N_pix))
            i_fr = ts_frame[i_ind]
            ts_ms[count]      = t_ms[i_fr]
            ts_vid[count,:,:] = vid_reach[:,:,i_fr]
            count = count + 1
        
        # plt.imshow(vid_reach[:,:,400])
    if ts_ms.ndim == 1:
        ts_vid = ((ts_vid - ts_vid.min()) * (1/(ts_vid.max() - ts_vid.min()) * 255)).astype('uint8')
        ts_vid = ts_vid[:,:,:,np.newaxis]
        ts_vid = np.repeat(ts_vid,3, axis = 3)
        
        ts_ms = ((ts_ms - ts_ms.min()) * (1/(ts_ms.max() - ts_ms.min()) * 255)).astype('uint8')
        ts_ms = ts_ms[:,np.newaxis]
    
        iPR_ms = ((iPR_ms - iPR_ms.min()) * (1/(iPR_ms.max() - iPR_ms.min()) * 255)).astype('uint8')
        iPR_ms = iPR_ms.T #[:,np.newaxis]
    
    # ts_cond = ts_ms
    ts_cond = iPR_ms
    
    
    # for i_n in np.arange(N_neur):
    #     iPR_ms[:,i_n] = np.gradient(iPR_ms[:,i_n])
    
    # iPR_tmp = iPR_ms[0:65,0]
    # iPR_slope_ms = np.gradient(iPR_tmp)
    # plt.plot(iPR_tmp)
    # plt.plot(iPR_slope_ms)
    # plt.plot(np.zeros(65))
    # plt.imshow()
    
    ## models
    
    
    def define_discriminator(in_shape=(N_pix,N_pix,3)):
        	
        # label input
        in_label = Input(shape=(np.size(ts_cond,1),))
        n_nodes = in_shape[0] * in_shape[1] 
        li = Dense(n_nodes)(in_label) 
        li = Reshape((in_shape[0], in_shape[1], 1))(li) 
        
        # image input
        in_image = Input(shape=in_shape) 
        # concat label as a channel
        merge = Concatenate()([in_image, li]) 
         
        fe = Conv2D(Conv_unit, (3,3), strides=(2,2), padding='same')(merge) 
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv2D(Conv_unit, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv2D(Conv_unit, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv2D(Conv_unit, (3,3), strides=(2,2), padding='same')(fe) 
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)  
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)  #Shape=1
           
        # define model
        model = Model([in_image, in_label], out_layer)
 
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
       
    
    # define generator model
    
    def define_generator(latent_dim):
        
        	# label input
        in_label = Input(shape=(np.size(ts_cond,1),)) 
        
        n_nodes = 8 * 8 
        li = Dense(n_nodes)(in_label) #1,64
        li = Reshape((8, 8, 1))(li)
    
        
        	# image generator input
        in_lat = Input(shape=(latent_dim,))  #Input of dimension 100
        
        n_nodes = Conv_unit * 8 * 8
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((8, 8, Conv_unit))(gen) 
        	# merge image gen and label input
        merge = Concatenate()([gen, li]) 
        gen = Conv2DTranspose(Conv_unit, (4,4), strides=(2,2), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2DTranspose(Conv_unit, (4,4), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2DTranspose(Conv_unit, (4,4), strides=(2,2), padding='same')(gen) 
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2DTranspose(Conv_unit, (4,4), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # output
        out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) 
        	# define model
        model = Model([in_lat, in_label], out_layer)
        return model   
    
    
    def define_gan(g_model, d_model):
    	d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
        
    
    	gen_noise, gen_label = g_model.input 
    	# get image output from the generator model
    	gen_output = g_model.output  
        
    	# generator image output and corresponding input label are inputs to discriminator
    	gan_output = d_model([gen_output, gen_label])
    	# define gan model as taking noise and label and outputting a classification
    	model = Model([gen_noise, gen_label], gan_output)
    	# compile model
    	opt = Adam(lr=0.0002, beta_1=0.5)
    	model.compile(loss='binary_crossentropy', optimizer=opt)
    	return model
    
    # load cifar images
    def load_real_samples():
    	# load dataset
        trainX = ts_vid 
        trainy = ts_cond
        # convert to floats and scale
        X = trainX.astype('float32')
    	# scale from [0,255] to [-1,1]
        X = (X - 127.5) / 127.5   #Generator uses tanh activation so rescale 
                                #original images to -1 to 1 to match the output of generator.
        return [X, trainy]
    
    def generate_real_samples(dataset, n_samples):
    	# split into images and labels
    	images, labels = dataset  
    	# choose random instances
    	ix = randint(0, images.shape[0], n_samples)
    	# select images and labels
    	X, labels = images[ix], labels[ix]
        
    	y = ones((n_samples, 1))  #Label=1 indicating they are real
    	return [X, labels], y
    
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples):
        # n_samples =100
        	# generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        	# reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        
        ind_rnd = np.arange(np.size(ts_cond,0))
        np.random.shuffle(ind_rnd)
        labels = ts_cond[ind_rnd[0:n_samples],:]
        return [z_input, labels]
    
    def generate_fake_samples(generator, latent_dim, n_samples):
    	# generate points in latent space
    	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    	# predict outputs
    	images = generator.predict([z_input, labels_input])
    	# create class labels
    	y = zeros((n_samples, 1))  #Label=0 indicating they are fake
    	return [images, labels_input], y
    
    
    def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=32): #128):
    	bat_per_epo = int(dataset[0].shape[0] / n_batch)
    	half_batch = int(n_batch / 2)  #the discriminator model is updated for a half batch of real samples 
                                #and a half batch of fake samples, combined a single batch. 
    	count = 0
    	train_loss  = zeros((n_epochs*bat_per_epo,7)) 
    	# manually enumerate epochs
    	for i in range(n_epochs):
    		# enumerate batches over the training set
    		for j in range(bat_per_epo):
    			start_time = time.time()
    			# i = 0 
                # j = 0
    
    			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
    
    			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                
    			# generate 'fake' examples
    			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    			# update discriminator model weights
    			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)  
                
    			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
                
    			y_gan = ones((n_batch, 1))
    
    			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
    			# Print losses on this batch
    			t_epoch = (time.time() - start_time)
    			print('rr%d, day%d, Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f, %.3fs/batch' %(rr_i, day_i, i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss, t_epoch))
    			train_loss[count,:] = [i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss, t_epoch]
    			count = count + 1
                
    			del X_real, X_fake, labels_real, d_loss_real, labels, d_loss_fake, z_input, labels_input, y_gan, g_loss
    			gc.collect()
    	# save the generator model
    	g_model.save(str_save_prefix + '_' + str(len(ts_ms)) + 'sample_' + str(N_pix) + 'pix_' + str(n_epochs) + 'epochs.h5')
    	np.save(str_save_prefix + '_' + str(len(ts_ms)) + 'sample_' + str(N_pix) + 'pix_' + str(n_epochs) + 'epochs_loss.npy', train_loss)
    #Train the GAN
    
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # load image data
    dataset = load_real_samples()
    # train model
    os.chdir(save_path)
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100) 
    
    ### save model

    n_epochs = 100
    model = load_model(str_save_prefix + '_' + str(len(ts_ms)) + 'sample_' + str(N_pix) + 'pix_' + str(n_epochs) + 'epochs.h5')
    
    #### save generated image
        
    labels = ts_cond
    latent_points, labels_tmp = generate_latent_points(100, len(labels))
    X_gan  = model.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    X_gan = (X_gan + 1) / 2.0
    X_gan = (X_gan*255).astype(np.uint8)
    
    os.chdir(save_path)
    np.save('rat' + str(rr_i) + '_day' + str(day_i) + '_GAN_iPR_images_' + str(N_pix) + 'pix_' + str(t_ms_incr) + 'msIncr.npy', X_gan)
    # plt.imshow(X_gan[100,:,:,:])


    ########################## plot single reach
    
    #################### plot GAN reach
    labels = ts_cond[0:N_iPR,:]
    latent_points, labels_tmp = generate_latent_points(100, len(labels))
    X  = model.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    X = (X*255).astype(np.uint8)
    t_ms_vec = np.arange(t_ms_seg[0],t_ms_seg[1]+1,t_ms_incr)
    for i in range(N_iPR-1):
        axes = plt.subplot(5, 10, i+1)
        axes.axis('off')
        X_tmp = X[i, :, :, 0]
        axes.imshow(X_tmp, cmap='gray')
        axes.set_title(str(t_ms_vec[i]) + 'ms')

    #################### plot real reach
    X_real = ts_vid[0:N_iPR,:,:,:]
    t_ms_vec = np.arange(t_ms_seg[0],t_ms_seg[1]+1,t_ms_incr)
    for i in range(N_iPR-1):
        axes = plt.subplot(5, 10, i+1)
        axes.axis('off')
        X_tmp = X_real[i, :, :, 0]
        axes.imshow(X_tmp, cmap='gray')
        axes.set_title(str(t_ms_vec[i]) + 'ms')       
