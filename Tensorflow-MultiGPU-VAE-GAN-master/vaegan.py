#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:28:26 2017

@author: cisa
"""

# Import all of our packages

import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from deconv import deconv2d
import IPython.display
import math
import tqdm # making loops prettier
import h5py # for reading our dataset
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import mnist_reader
from function import encoder, generator, discriminator, inference, loss, average_gradients#, plot_network_output
get_ipython().magic(u'matplotlib inline')

dim1 = 28 # first dimension of input data
dim2 = 28 # second dimension of input data
dim3 = 1 # third dimension of input data (colors)
batch_size = 32 # size of batches to use (per GPU)
hidden_size = 2048 # size of hidden (z) layer to use
num_examples = 60000 # how many examples are in your training set
num_epochs = 10000 # number of epochs to run
### we can train our different networks  with different learning rates if we want to
e_learning_rate = 5e-4
g_learning_rate = 5e-4
d_learning_rate = 5e-4

gpus = [0] # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use

x_train0,y_train = mnist_reader.load_mnist('asset/data/data/mnist',kind='train')
x_test0,y_test = mnist_reader.load_mnist('asset/data/data/mnist',kind='t10k')
x_train = np.reshape(x_train0,(60000,28*28))
# x_train = x_train / 255.0
x_test0 = np.reshape(x_test0,(10000,28*28))
# x_test = x_test0 / 255.0

refined_label = [0, 1, 2, 3, 4]#, 5, 6, 7, 8, 9]
majority_label = [5,6,7,8,9]
num_train_per_label = [50,50,50,50,50]#,2500,2500,2500,2500,2500]
train_refined_label_idx = np.array([], dtype = np.uint8)
test_refined_label_idx = np.array([], dtype = np.uint8)
for idx, label_value in enumerate(refined_label):
    refined_one_label_idx = np.where( y_train == label_value )[0][:num_train_per_label[idx]]
    train_refined_label_idx = np.append( train_refined_label_idx,  refined_one_label_idx)
x = x_train[train_refined_label_idx, :]
y = y_train[train_refined_label_idx]

# Just taking a look and making sure everything works
plt.imshow(np.reshape(x[50].astype(np.float32), (28,28)), cmap='gray')


def create_image(im):
    return np.reshape(im,(dim1,dim2))

#cm = plt.cm.hot
#test_face = x[0].reshape(dim1,dim2,dim3)
#fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(20,8))
#ax[0].imshow(create_image(test_face), interpolation='nearest')
#ax[1].imshow(create_image(test_face)[:,:,0], interpolation='nearest', cmap=cm)
#ax[2].imshow(create_image(test_face)[:,:,1], interpolation='nearest', cmap=cm)
#ax[3].imshow(create_image(test_face)[:,:,2], interpolation='nearest', cmap=cm)

def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(x), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx+batch_size]
            images_batch = x[cur_idxs]
            #images_batch = images_batch.astype("float32")
            labels_batch = y[cur_idxs]
            yield images_batch, labels_batch

            
iter_ = data_iterator()
##########################################
def plot_network_output():
    """ Just plots the output of the network, error, reconstructions, etc
    """
    random_x, recon_z, all_d= sess.run((x_p, z_x_mean, d_x_p), {all_input: example_data})
    top_d = np.argsort(np.squeeze(all_d))
    recon_x = sess.run((x_tilde), {z_x: recon_z})
    examples = 8
    random_x = np.squeeze(random_x)
    recon_x = np.squeeze(recon_x)
    random_x = random_x[top_d]
   
    fig, ax = plt.subplots(nrows=3,ncols=examples, figsize=(28,6))
    for i in xrange(examples):
        ax[(0,i)].imshow(create_image(random_x[i]), cmap='gray', interpolation='nearest')
        ax[(1,i)].imshow(create_image(recon_x[i]), cmap='gray', interpolation='nearest')
        ax[(2,i)].imshow(create_image(example_data[i + (num_gpus-1)*batch_size]), cmap='gray', interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        ax[(2,i)].axis('off')
    fig.suptitle('Top: random points in z space | Bottom: inputs | Middle: reconstructions')
    plt.show()
    #fig.savefig(''.join(['imgs/test_',str(epoch).zfill(4),'.png']),dpi=100)
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10), linewidth = 4)
    KL_plt, = plt.semilogy((KL_loss_list), linewidth = 4, ls='-', color='r', alpha = .5, label='KL')
    D_plt, = plt.semilogy((D_loss_list),linewidth = 4, ls='-', color='b',alpha = .5, label='D')
    G_plt, = plt.semilogy((G_loss_list),linewidth = 4, ls='-', color='k',alpha = .5, label='G')
    SSE_plt, = plt.semilogy((SSE_loss_list),linewidth = 4,ls='-', color='g',alpha = .5, label='SSE')
    LL_plt, = plt.semilogy((LL_loss_list),linewidth = 4,ls='-', color='m',alpha = .5, label='LL')
    
    axes = plt.gca()
    leg = plt.legend(handles=[KL_plt, D_plt, G_plt, SSE_plt, LL_plt], fontsize=20)
    leg.get_frame().set_alpha(0.5)
    plt.show()
#############################
graph = tf.Graph()


##########################################################
#pic = create_image(random_x[2])
#plt.imshow(np.reshape(pic.astype(np.float32),(28,28)), cmap='gray')
#plt.axis('off')
#plt.show()
##########################################################
# Make lists to save the losses to 
# You should probably just be using tensorboard to do any visualization(or just use tensorboard...)
G_loss_list = [] 
D_loss_list = [] 
SSE_loss_list = []
KL_loss_list = []
LL_loss_list = []
dxp_list = []
dx_list = []


with graph.as_default():
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count number of train calls
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)


    # different optimizers are needed for different learning rates (using the same learning rate seems to work fine though)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
    opt_G = tf.train.AdamOptimizer(lr_G, epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)
    
  
with graph.as_default():

    # These are the lists of gradients for each tower 
    tower_grads_e = []
    tower_grads_g = []
    tower_grads_d = []

    all_input = tf.placeholder(tf.float32, [batch_size*num_gpus, dim1*dim2*dim3])
    KL_param = tf.placeholder(tf.float32)
    LL_param = tf.placeholder(tf.float32)
    G_param = tf.placeholder(tf.float32)


    # Define the network for each GPU
    for i in xrange(num_gpus):
          with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    with tf.variable_scope(tf.get_variable_scope()) as scope:  
                        # grab this portion of the input
                        next_batch = all_input[i*batch_size:(i+1)*batch_size,:]
    
                        # Construct the model
                        z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p, d_x_tilde = inference(next_batch)
    
                        # Calculate the loss for this tower   
                        SSE_loss, KL_loss, D_loss, G_loss, LL_loss = loss(next_batch, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, d_x_tilde, dim1, dim2, dim3)
    
                        # specify loss to parameters
                        params = tf.trainable_variables()
                        E_params = [i for i in params if 'enc' in i.name]
                        G_params = [i for i in params if 'gen' in i.name]
                        D_params = [i for i in params if 'dis' in i.name]
    
                        # Calculate the losses specific to encoder, generator, decoder
                        L_e = tf.clip_by_value(KL_loss*KL_param + LL_loss, -100, 100)
                        L_g = tf.clip_by_value(LL_loss*LL_param-D_loss*G_param, -100, 100)
                        L_d = tf.clip_by_value(D_loss, -100, 100)
    
    
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads_e = opt_E.compute_gradients(L_e, var_list = E_params)
                    grads_g = opt_G.compute_gradients(L_g, var_list = G_params)
                    grads_d = opt_D.compute_gradients(L_d, var_list = D_params)

                    # Keep track of the gradients across all towers.
                    tower_grads_e.append(grads_e)
                    tower_grads_g.append(grads_g)
                    tower_grads_d.append(grads_d)

#with tf.variable_scope(tf.get_variable_scope()) as scope:  
with graph.as_default():
    # Average the gradients
    grads_e = average_gradients(tower_grads_e)
    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    # apply the gradients with our optimizers
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_G = opt_G.apply_gradients(grads_g, global_step=global_step)
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)
    

with graph.as_default():

    # Start the Session
    init = tf.initialize_all_variables()
    saver = tf.train.Saver() # initialize network saver
    sess = tf.InteractiveSession(graph=graph,config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(init)

example_data, _ = iter_.next()
np.shape(example_data)
epoch=0
##########################################################################################################
#tf.train.Saver.restore(saver, sess, 'asset/train0055.tfmod')
#########################################################################################################

def sigmoid(x,shift,mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + math.exp(-(x+shift)*mult))

fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(18,4))
plt.plot(np.arange(0,1,.01), [sigmoid(i/100.,-.5,10) for i in range(100)])
ax.set_xlabel('Mean of Discriminator(Real) or Discriminator(Fake)')
ax.set_ylabel('Multiplier for learning rate')
plt.title('Squashing the Learning Rate to balance Discrim/Gen network performance')

# how many batches are in an epoch
total_batch = int(np.floor(num_examples / batch_size*num_gpus)) 
# We balance of generator and discriminators learning rate by using a sigmoid function,
#  encouraging the generator and discriminator be about equal
d_real = .5
d_fake = .5

while epoch < num_epochs:    
    for i in tqdm.tqdm(range(total_batch)):
        iter_ = data_iterator()
        # balence gen and descrim
        e_current_lr = e_learning_rate*sigmoid(np.mean(d_real),-.5,15)
        g_current_lr = g_learning_rate*sigmoid(np.mean(d_real),-.5,15)
        d_current_lr = d_learning_rate*sigmoid(np.mean(d_fake),-.5,15)
        next_batches, _ = iter_.next()

        _, _, _, D_err, G_err, KL_err, SSE_err, LL_err, d_fake,d_real = sess.run([
                train_E, train_G, train_D,
                D_loss, G_loss, KL_loss, SSE_loss, LL_loss,
                d_x_p, d_x,

            ],
                                        {
                lr_E: e_current_lr,
                lr_G: g_current_lr,
                lr_D: d_current_lr,
                all_input: next_batches,
                KL_param: 1,
                G_param: 1,
                LL_param: 1
            }
       )
        #KL_err= SSE_err= LL_err = 1
        # Save our lists
        dxp_list.append(d_fake)
        dx_list.append(d_real)
        G_loss_list.append(G_err)
        D_loss_list.append(D_err)
        KL_loss_list.append(KL_err)
        SSE_loss_list.append(SSE_err)
        LL_loss_list.append(LL_err)
    
        if i%300 == 0:
            # print display network output
            IPython.display.clear_output()
            print('Epoch: '+str(epoch))
            plot_network_output()
  
    # save network
    saver.save(sess,''.join(['asset/train',str(epoch).zfill(4),'.tfmod']))
    epoch +=1