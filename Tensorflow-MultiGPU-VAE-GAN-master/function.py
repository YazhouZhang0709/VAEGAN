#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:35:34 2017

@author: cisa
"""


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
 
get_ipython().magic(u'matplotlib inline')

dim1 = 28 # first dimension of input data
dim2 = 28 # second dimension of input data
dim3 = 1 # third dimension of input data (colors)
batch_size = 32 # size of batches to use (per GPU)
hidden_size = 2048 # size of hidden (z) layer to use
num_examples = 60000 # how many examples are in your training set
num_epochs = 10000 # number of epochs to run
### we can train our different networks  with different learning rates if we want to
e_learning_rate = 1e-3
g_learning_rate = 1e-3
d_learning_rate = 1e-3

gpus = [1] # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use
###############################encoder generator discriminator###############################
def encoder(X):
    '''Create encoder network.
    Args:
        x: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
            # The transformation is parametrized and can be learned.
            # returns network output, mean, setd
    '''
    lay_end = (pt.wrap(X).
            reshape([batch_size, dim1, dim2, dim3]). 
            conv2d(4, 64, stride=2). 
            conv2d(4, 128, stride=2). 
#             conv2d(5, 256, stride=2). 
            flatten())
    z_mean = lay_end.fully_connected(hidden_size, activation_fn=None)
    z_log_sigma_sq = lay_end.fully_connected(hidden_size, activation_fn=None)
    return z_mean, z_log_sigma_sq


def generator(Z):
    '''Create generator network.
        If input tensor is provided then decodes it, otherwise samples from 
        a sampled vector.
    Args:
        x: a batch of vectors to decode
    Returns:
        A tensor that expresses the generator network
    '''
    return (pt.wrap(Z).
            fully_connected(7*7*128).reshape([batch_size, 7, 7, 128]). #(128, 4 4, 256)
            deconv2d(4, 64, stride=2).
#             deconv2d(4, 1, stride=2).
#             deconv2d(5, 32, stride=2).
            deconv2d(1, dim3, stride=2, activation_fn=tf.sigmoid).
            flatten()
           )



def discriminator(D_I):
    ''' A encodes
    Create a network that discriminates between images from a dataset and
    generated ones.
    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    ''' 
    descrim_conv =  (pt.wrap(D_I). # This is what we're descriminating
            reshape([batch_size, dim1, dim2, dim3]).
            conv2d(4, 64, stride=2).
            conv2d(4, 128, stride=2).
#             conv2d(5, 256, stride=2).
#             conv2d(5, 256, stride=2).
            flatten()
            )
    lth_layer= descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)# this is the lth layer       
    D =lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid) # this is the actual discrimination
    return D, lth_layer
###############################encoder generator discriminator###############################

def inference(x):
    """
    Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
    """
    z_p =  tf.random_normal((batch_size, hidden_size), 0, 1) # normal dist for GAN
    eps = tf.random_normal((batch_size, hidden_size), 0, 1) # normal dist for VAE

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
        
        with tf.variable_scope("enc"):         
                z_x_mean, z_x_log_sigma_sq = encoder(x) # get z from the input      
        with tf.variable_scope("gen"):
            z_x = tf.add(z_x_mean, 
                tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps)) # grab our actual z
            x_tilde = generator(z_x)  
        with tf.variable_scope("dis"):   
            d_x_tilde, l_x_tilde = discriminator(x_tilde)
        with tf.variable_scope("gen", reuse=True):         
            x_p = generator(z_p)    
        with tf.variable_scope("dis", reuse=True):
            d_x, l_x = discriminator(x)  # positive examples              
        with tf.variable_scope("dis", reuse=True):
            d_x_p, _ = discriminator(x_p)  
        return z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p, d_x_tilde

def loss(x, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, d_x_tilde, dim1, dim2, dim3):
    """
    Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
    SSE_loss = tf.reduce_mean(tf.square(x - x_tilde)) # This is what a normal VAE uses

    # We clip gradients of KL divergence to prevent NANs
    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) 
                                   - tf.square(tf.clip_by_value(z_x_mean, -10.0, 10.0) ) 
                                   - tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) ), 1))/dim1/dim2/dim3
    # Discriminator Loss
    D_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x,1e-5,1.0)) + 
                                    tf.log(tf.clip_by_value(1.0 - d_x_p,1e-5,1.0))+ tf.log(tf.clip_by_value(1.0 - d_x_tilde,1e-5,1.0)) ))
    # Generator Loss    
    G_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x_p,1e-5,1.0))))# + 
                                    #tf.log(tf.clip_by_value(1.0 - d_x,1e-5,1.0))))
    # Lth Layer Loss - the 'learned similarity measure'  
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde))/dim1/dim2/dim3
    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
     
     
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        #np.asarray(grads, dtype=np.int32)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

